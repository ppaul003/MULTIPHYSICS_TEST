#include "kernel.h"
#include "TheTesseractEM.h"

#include <GL/freeglut.h>
#include <algorithm>
#include <cstdio>
#include <cmath>

#include "CameraEM.h"
#include "rendererEM_Euclid.h"

using namespace std;

bool Tesseract::initialize(WorkspaceServices services) {
    m_services = services;

    if (!m_diagnosticIdle.initialize(m_services)) {
        printf("[Tesseract] ERROR: DiagnosticIdle initialization failed.\n");
        return false;
    }

    if (!m_particleSimWorkspace.initialize(m_services)) {
        printf("[Tesseract] ERROR: ParticleSimWorkspace initialization failed.\n");
        return false;
    }

    if (!m_atomicParticlesSimWorkspace.initialize(m_services)) {
        printf("[Tesseract] ERROR: ATOMIC_PARTICLES initialization failed.\n");
        return false;
    }

    if (!m_multiPhySim.initialize(m_services)) {
        printf("[Tesseract] ERROR: MULTIPHYSICS_SIM initialization failed.\n");
        return false;
    }

    m_activeWorkspace = &m_diagnosticIdle;
    m_activeWorkspace->enter(m_services);
    printf("[Tesseract] Workspace socket initialized.\n");
    printf("[Tesseract] Active workspace: DIAGNOSTIC_IDLE\n");
    return true;
}

void Tesseract::shutdown() {
    if (m_activeWorkspace) m_activeWorkspace->exit(m_services);
    m_atomicParticlesSimWorkspace.releaseFieldResources();
    m_activeWorkspace = nullptr;
}

void Tesseract::update(const WorkspaceFrameContext& frame) {
    if (boxResizeActive()) {

        const int previousBox = m_services.renderer->getSimBoxSize();

        m_diagnosticIdle.update(frame, m_services);

        if (m_services.renderer->getSimBoxSize() != previousBox) {

            m_particleSimWorkspace.syncSimulationDomain(m_services);
            m_atomicParticlesSimWorkspace.syncSimulationDomain(m_services);
            m_multiPhySim.syncSimulationDomain(m_services);
        }
        return;
    }
    processNavigationRequest();

    if (domainTransitionActive()) {
        updateDomainTransition(frame);
        return;
    }

    synchronizeActiveCartridge();
    if (m_activeWorkspace)
        m_activeWorkspace->update(frame, m_services);
}

void Tesseract::render(const WorkspaceFrameContext& frame) {
    if (domainTransitionActive()) {
        switch (m_domainTransitionPhase) {
        case DomainTransitionPhase::ENTER_DOMAIN_VISUAL:
        case DomainTransitionPhase::ENTER_DOMAIN_READY:
        case DomainTransitionPhase::ENTER_CAMERA:
        case DomainTransitionPhase::EXIT_DOMAIN_VISUAL:
            m_diagnosticIdle.render(frame, m_services);
            return;

        case DomainTransitionPhase::EXIT_CAMERA:
            if (m_activeWorkspace)
                m_activeWorkspace->render(frame, m_services);
            return;

        case DomainTransitionPhase::NONE:
        default:
            break;
        }
    }

    synchronizeActiveCartridge();
    if (m_activeWorkspace)
        m_activeWorkspace->render(frame, m_services);
}

bool Tesseract::handleInput(const WorkspaceInputEvent& event) {
    if (domainTransitionActive() || boxResizeActive()) return true;

    if (!m_activeWorkspace) return false;

    const bool handled = m_activeWorkspace->handleInput(event, m_services);
    processNavigationRequest();

    if (!domainTransitionActive())
        synchronizeActiveCartridge();

    return handled;
}

bool Tesseract::handleInputRelease(const WorkspaceInputEvent& event) {
    if (boxResizeActive()) return true;
    return m_activeWorkspace &&
        m_activeWorkspace->handleInputRelease(event, m_services);
}

bool Tesseract::handlePointerInput(const WorkspacePointerEvent& event) {
    if (domainTransitionActive() || boxResizeActive()) return true;
    return m_activeWorkspace &&
        m_activeWorkspace->handlePointerInput(event, m_services);
}

void Tesseract::cancelInput() {
    if (m_activeWorkspace) m_activeWorkspace->cancelInput(m_services);
}

void Tesseract::renderOverlay(const WorkspaceFrameContext& frame) {
    if (!domainTransitionActive() && m_activeWorkspace)
        m_activeWorkspace->renderOverlay(frame, m_services);
}

WorkspaceMenuPresentation Tesseract::menu() const {
    return m_activeWorkspace
        ? m_activeWorkspace->buildMenu()
        : WorkspaceMenuPresentation{};
}

bool Tesseract::handleMenuCommand(int command) {
    if (domainTransitionActive() || boxResizeActive() || !m_activeWorkspace) return false;
    const bool handled = m_activeWorkspace->handleMenuCommand(command, m_services);
    processNavigationRequest();
    if (!domainTransitionActive()) synchronizeActiveCartridge();
    return handled;
}

WorkspacePresentation Tesseract::presentation() const {
    using Domain = TheArbiter::WorkspaceDomain;
    using Phase = DomainTransitionPhase;

    if (!m_activeWorkspace) {
        WorkspacePresentation p;
        p.panelVisible = true;
        p.workspaceName = "SIMULATION MODULE TEST & DEBUG";
        p.layerLabel = "NO ACTIVE WORKSPACE";
        p.statusLine = "WORKSPACE SOCKET OFFLINE";
        return p;
    }

    if (m_transitionDomain == Domain::MULPHY_SIM) {
        switch (m_domainTransitionPhase) {
        case Phase::ENTER_DOMAIN_VISUAL: {
            WorkspacePresentation p = m_diagnosticIdle.buildPresentation();
            p.statusLine = "AUTO: Transitioning To MULTIPHYSICS...";
            p.statusTone = WorkspaceStatusTone::Transition;
            p.statusBlink = true;
            p.frameTone = WorkspaceStatusTone::Transition;
            p.frameBlink = true;
            return p;
        }
        case Phase::ENTER_DOMAIN_READY: {
            WorkspacePresentation p = m_diagnosticIdle.buildPresentation();
            p.statusLine = "READY: MULTIPHYSICS Setup Complete.";
            p.statusTone = WorkspaceStatusTone::Ready;
            p.statusBlink = false;
            p.frameTone = WorkspaceStatusTone::Ready;
            p.frameBlink = false;
            return p;
        }
        case Phase::ENTER_CAMERA: {

            WorkspacePresentation p = m_diagnosticIdle.buildPresentation();
            p.statusLine = "READY: Entering MULTIPHYSICS workspace...";
            p.statusTone = WorkspaceStatusTone::Ready;
            p.frameTone = WorkspaceStatusTone::Ready;
            p.frameBlink = false;
            return p;
        }
        default:
            break;
        }
    }

    return m_activeWorkspace->buildPresentation();
}

void Tesseract::processNavigationRequest() {
    if (!m_services.arbiter) return;
    if (domainTransitionActive() || boxResizeActive()) return;
    if (!m_services.arbiter->hasNavigationRequest()) return;

    const TheArbiter::NavigationRequest request =
        m_services.arbiter->takeNavigationRequest();

    using Request = TheArbiter::NavigationRequestType;
    using Domain = TheArbiter::WorkspaceDomain;

    switch (request.type) {
    case Request::ENTER_DOMAIN:
        // Only the supported test-shell configuration can request entry.
        if (request.domain != Domain::MULPHY_SIM ||
            !m_services.arbiter->isGlobalShell() ||
            m_services.arbiter->getUnitMeasurement() != TheArbiter::UnitMeasurement::METRIC)
            return;

        m_transitionDomain = request.domain;
        m_domainTransitionPhase = DomainTransitionPhase::ENTER_DOMAIN_VISUAL;

        m_diagnosticIdle.beginMulphyEnterTransition();
        return;

    case Request::RETURN_GLOBAL_SHELL:
        m_transitionDomain = request.domain;

        if (request.domain == Domain::MULPHY_SIM) {
            m_domainTransitionPhase = DomainTransitionPhase::EXIT_CAMERA;
            if (m_services.camera)
                m_services.camera->beginTransitionToMenu(kCameraTransitionDuration, domainCameraScale());
            else {
                m_domainTransitionPhase = DomainTransitionPhase::EXIT_DOMAIN_VISUAL;
                m_diagnosticIdle.beginMulphyReturnTransition();
            }
            return;
        }
        return;

    case Request::NONE:
    default:
        return;
    }
}

void Tesseract::updateDomainTransition(const WorkspaceFrameContext& frame) {
    if (!m_services.arbiter || !domainTransitionActive()) return;

    using Domain = TheArbiter::WorkspaceDomain;
    using Layer = TheArbiter::ApplicationLayer;
    using Phase = DomainTransitionPhase;

    switch (m_domainTransitionPhase) {
    case Phase::ENTER_DOMAIN_VISUAL:
        m_diagnosticIdle.update(frame, m_services);

        if (!m_diagnosticIdle.mulphyEnterVisualComplete()) return;

        m_domainTransitionPhase = Phase::ENTER_DOMAIN_READY;
        m_transitionPhaseElapsed = 0.0f;
        return;

    case Phase::ENTER_DOMAIN_READY:
        m_diagnosticIdle.update(frame, m_services);
        m_transitionPhaseElapsed += frame.deltaTime;
        if (m_transitionPhaseElapsed < kReadyHoldDuration) return;

        m_transitionPhaseElapsed = 0.0f;

        m_domainTransitionPhase = Phase::ENTER_CAMERA;
        if (m_services.camera)
            m_services.camera->beginTransitionToStandard3D(kCameraTransitionDuration, domainCameraScale());
        return;

    case Phase::ENTER_CAMERA:

        m_diagnosticIdle.update(frame, m_services);

        if (m_services.camera) {

            m_services.camera->updatePoseTransition(frame.deltaTime);

            if (m_services.camera->poseTransitionActive()) return;
            m_services.camera->setBehaviorMode(CameraProcessor::CAM_STANDARD_3D, domainCameraScale());
        }

        m_services.arbiter->setActiveWorkspace(TheArbiter::WorkspaceId::PARTICLE_SIM);
        m_services.arbiter->setApplicationLayer(Layer::DOMAIN_SELECTION);

        m_domainTransitionPhase = Phase::NONE;
        m_transitionDomain = Domain::NONE;
        synchronizeActiveCartridge();

        return;

    case Phase::EXIT_CAMERA:
        if (m_activeWorkspace)
            m_activeWorkspace->update(frame, m_services);

        if (m_services.camera) {

            m_services.camera->updatePoseTransition(frame.deltaTime);
            if (m_services.camera->poseTransitionActive()) return;
        }

        m_domainTransitionPhase = Phase::EXIT_DOMAIN_VISUAL;
        m_diagnosticIdle.beginMulphyReturnTransition();
        return;

    case Phase::EXIT_DOMAIN_VISUAL:
        m_diagnosticIdle.update(frame, m_services);

        if (!m_diagnosticIdle.mulphyReturnVisualComplete()) return;

        m_services.arbiter->setApplicationLayer(Layer::GLOBAL_SHELL);
        m_services.arbiter->setActiveWorkspace(TheArbiter::WorkspaceId::DIAGNOSTIC);

        if (m_services.camera)
            m_services.camera->setBehaviorMode(CameraProcessor::CAM_MENU_PREVIEW, domainCameraScale());

        m_domainTransitionPhase = Phase::NONE;
        m_transitionDomain = Domain::NONE;
        synchronizeActiveCartridge();
        return;

    case Phase::NONE:
    default:
        return;
    }
}

float Tesseract::domainCameraScale() const {
    return m_services.renderer ? static_cast<float>(m_services.renderer->getSimBoxSize()) / 4.0f : 1.0f;
}

void Tesseract::synchronizeActiveCartridge() {
    if (!m_services.arbiter) return;

    IWorkspace* desired = &m_diagnosticIdle;
    const char* desiredName = "DIAGNOSTIC_IDLE";

    if (!m_services.arbiter->isGlobalShell()) {

        using Domain = TheArbiter::WorkspaceDomain;
        using Workspace = TheArbiter::WorkspaceId;

        if (m_services.arbiter->getWorkspaceDomain() == Domain::MULPHY_SIM) {

            switch (m_services.arbiter->getActiveWorkspace()) {

            case Workspace::PARTICLE_SIM:
                desired = &m_particleSimWorkspace;
                desiredName = "PARTICLE_SIMULATION";
                break;

                case Workspace::ATOMIC_PARTICLES:
                desired = &m_atomicParticlesSimWorkspace;
                desiredName = "ATOMIC_PARTICLES";
                break;

            case Workspace::MULTIPHYSICS_SIM:
                desired = &m_multiPhySim;
                desiredName = "MULTIPHYSICS_SIM";
                break;

            default:
                break;
            }
        }
    }

    activateCartridge(desired, desiredName);
}

void Tesseract::activateCartridge(IWorkspace* workspace, const char* name) {
    if (!workspace || workspace == m_activeWorkspace) return;

    if (m_activeWorkspace)
        m_activeWorkspace->exit(m_services);

    m_activeWorkspace = workspace;
    m_activeWorkspace->enter(m_services);

    std::printf("[Tesseract] Active workspace: %s\n",
        name ? name : "UNKNOWN");
}
