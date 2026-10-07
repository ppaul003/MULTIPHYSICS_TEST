#ifndef NDMSM_TESSERACT_EM_H
#define NDMSM_TESSERACT_EM_H

#include <GL/glew.h>
#include <cstddef>
#include <vector>
#include <vector_types.h>

#include "IWorkspaceEM.h"
#include "WorkspaceContextEM.h"
#include "WorkspaceInputEM.h"
#include "WorkspacePresentationEM.h"

#include "TheArbiterEM.h"

#include "DiagnosticIdleEM.h"
#include "ParticleSimWorkspace.h"
#include "atomicParticlesSimWorkspace.h"
#include "multiPhysicsSimWorkspace.h"


class Tesseract {
public:
    enum class DomainTransitionPhase {
        NONE = 0,

        ENTER_DOMAIN_VISUAL,
        ENTER_DOMAIN_READY,
        ENTER_CAMERA,

        EXIT_CAMERA,
        EXIT_DOMAIN_VISUAL
    };

    bool initialize(WorkspaceServices services);
    void shutdown();

    void update(const WorkspaceFrameContext& frame);
    void render(const WorkspaceFrameContext& frame);
    bool handleInput(const WorkspaceInputEvent& event);
    bool handleInputRelease(const WorkspaceInputEvent& event);
    bool handlePointerInput(const WorkspacePointerEvent& event);
    void cancelInput();
    void renderOverlay(const WorkspaceFrameContext& frame);

    WorkspacePresentation presentation() const;
    WorkspaceMenuPresentation menu() const;
    bool handleMenuCommand(int command);

    void processNavigationRequest();
    void updateDomainTransition(const WorkspaceFrameContext& frame);

    bool domainTransitionActive() const {
        return m_domainTransitionPhase != DomainTransitionPhase::NONE;
    }
    bool boxResizeActive() const { return m_diagnosticIdle.boxResizeActive(); }
    DiagnosticIdle::BoxResizeState boxResizeState() const { return m_diagnosticIdle.boxResizeState(); }
    SimulationDomainState particleDomainState() const { return m_particleSimWorkspace.simulationDomainState(); }
    SimulationDomainState atomicDomainState() const { return m_atomicParticlesSimWorkspace.simulationDomainState(); }
    SimulationDomainState multiphysicsDomainState() const { return m_multiPhySim.simulationDomainState(); }

private:
    void synchronizeActiveCartridge();
    void activateCartridge(IWorkspace* workspace, const char* name);
    float domainCameraScale() const;

    static constexpr float kCameraTransitionDuration = 0.75f;
    static constexpr float kReadyHoldDuration = 0.35f;

    WorkspaceServices m_services;
    DiagnosticIdle m_diagnosticIdle;

    ParticleSimWorkspace m_particleSimWorkspace;
    AtomicParticlesSimWorkspace m_atomicParticlesSimWorkspace;
    MultiPhysicsSimWorkspace m_multiPhySim;

    IWorkspace* m_activeWorkspace = nullptr;

    DomainTransitionPhase m_domainTransitionPhase = DomainTransitionPhase::NONE;
    TheArbiter::WorkspaceDomain m_transitionDomain = TheArbiter::WorkspaceDomain::NONE;
    float m_transitionPhaseElapsed = 0.0f;
};

#endif
