#include "multiPhysicsSimWorkspace.h"
#include "CameraEM.h"
#include "rendererEM_Euclid.h"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <sstream>

namespace {
    const char* speciesName(DebugProjectileSpecies species) {
        switch (species) {
        case DebugProjectileSpecies::ArgonIon: return "ARGON_ION";
        case DebugProjectileSpecies::ArgonNeutral: return "ARGON_NEUTRAL";
        default: return "ELECTRON";
        }
    }
    std::string magnitudeText(double magnitude) {
        std::ostringstream out;
        out << std::scientific << std::setprecision(2) << magnitude;
        return out.str();
    }
}

bool MultiPhysicsSimWorkspace::layer3Active() const {
    return m_active && m_arbiter &&
        m_arbiter->getApplicationLayer() == TheArbiter::ApplicationLayer::ACTIVE_WORKSPACE &&
        m_arbiter->getActiveWorkspace() == TheArbiter::WorkspaceId::MULTIPHYSICS_SIM;
}

void MultiPhysicsSimWorkspace::initializeFields() {
    m_electronDensity.initialize(m_baseVoxelGrid);
    m_electronTemperature.initialize(m_baseVoxelGrid);
    m_chargeDensity.initialize(m_baseVoxelGrid);
    m_electricField.initialize(m_baseVoxelGrid);
    m_magneticField.initialize(m_baseVoxelGrid);
    m_currentDensity.initialize(m_baseVoxelGrid);
    m_curlMagneticField.initialize(m_baseVoxelGrid);
}

void MultiPhysicsSimWorkspace::clearFieldDebug() {
    m_debugElectrodynamics.clear();
    m_electronDensity.clear();
    m_electronTemperature.clear();
    m_chargeDensity.clear();
    m_electricField.clear();
    m_magneticField.clear();
    m_currentDensity.clear();
    m_curlMagneticField.clear();
    m_testFireMode = m_fireClickCaptured = m_freeLookDragging = false;
    std::fill(std::begin(m_freeMovementKeys), std::end(m_freeMovementKeys), false);
    m_vectorView = VectorView::Off;
    m_scalarView = ScalarView::Off;
    m_debugSpecies = DebugProjectileSpecies::Electron;
    m_debugNotice.clear();
    m_debugNoticeSeconds = 0.0f;
}

void MultiPhysicsSimWorkspace::refreshDiagnosticFields() {
    // Only explicitly fired debug projectiles source these diagnostics.
    // The production CUDA population is not a self-consistent PIC plasma.
    m_debugElectrodynamics.populateFields(m_baseVoxelGrid, m_electricField,
        m_magneticField, m_currentDensity, m_chargeDensity);
    computeCurl(m_magneticField, m_curlMagneticField);
}

void MultiPhysicsSimWorkspace::updateFieldDebug(
    const WorkspaceFrameContext& frame, WorkspaceServices& services) {
    m_debugNoticeSeconds = (std::max)(0.0f, m_debugNoticeSeconds - frame.deltaTime);
    if (m_layer3CameraView == Layer3CameraView::Free && services.camera) {
        services.camera->moveFree(
            static_cast<float>(m_freeMovementKeys[0]) - static_cast<float>(m_freeMovementKeys[1]),
            static_cast<float>(m_freeMovementKeys[3]) - static_cast<float>(m_freeMovementKeys[2]),
            frame.deltaTime);
    }
    if (!m_runtimeEnabled || m_paused || frame.deltaTime <= 0.0f ||
        m_debugElectrodynamics.projectiles().empty()) return;
    m_debugElectrodynamics.update(frame.deltaTime, m_baseVoxelGrid);
    refreshDiagnosticFields();
}

const VectorField3D* MultiPhysicsSimWorkspace::selectedVectorField() const {
    switch (m_vectorView) {
    case VectorView::Electric: return &m_electricField;
    case VectorView::Magnetic: return &m_magneticField;
    case VectorView::Current: return &m_currentDensity;
    case VectorView::CurlB: return &m_curlMagneticField;
    default: return nullptr;
    }
}

const ScalarField3D* MultiPhysicsSimWorkspace::selectedScalarField() const {
    switch (m_scalarView) {
    case ScalarView::ElectronDensity: return &m_electronDensity;
    case ScalarView::ElectronTemperature: return &m_electronTemperature;
    case ScalarView::ChargeDensity: return &m_chargeDensity;
    default: return nullptr;
    }
}

const char* MultiPhysicsSimWorkspace::vectorViewName() const {
    switch (m_vectorView) {
    case VectorView::Electric: return "ELECTRIC_FIELD [V/m]";
    case VectorView::Magnetic: return "MAGNETIC_FIELD [T]";
    case VectorView::Current: return "CURRENT_DENSITY [A/m^2]";
    case VectorView::CurlB: return "CURL_B [T/m]";
    default: return "OFF";
    }
}

const char* MultiPhysicsSimWorkspace::scalarViewName() const {
    switch (m_scalarView) {
    case ScalarView::ElectronDensity: return "ELECTRON_DENSITY [m^-3] (ZERO PLACEHOLDER)";
    case ScalarView::ElectronTemperature: return "ELECTRON_TEMPERATURE [eV] (ZERO PLACEHOLDER)";
    case ScalarView::ChargeDensity: return "CHARGE_DENSITY [C/m^3]";
    default: return "OFF";
    }
}

void MultiPhysicsSimWorkspace::renderFieldDebug(WorkspaceServices& services) {
    if (!services.renderer) return;
    FieldDebugRenderer::drawProjectiles(*services.renderer, m_debugElectrodynamics.projectiles());
    if (const auto* field = selectedVectorField())
        FieldDebugRenderer::drawVector(*field, m_vectorRenderSettings);
    if (const auto* field = selectedScalarField()) FieldDebugRenderer::drawScalar(*field);
}

bool MultiPhysicsSimWorkspace::handleFieldDebugKey(
    const WorkspaceInputEvent& input, WorkspaceServices& services) {
    const unsigned char key = static_cast<unsigned char>(std::tolower(input.rawKey));
    if (key != 'f' && key != 'v' && key != 'b' && key != 'c' && key != 'g' &&
        key != '1' && key != '2' && key != '3') return false;
    if (input.repeated) return true;
    switch (key) {
    case 'f':
        cancelInput(services);
        m_testFireMode = !m_testFireMode;
        if (m_testFireMode && m_vectorView == VectorView::Off) m_vectorView = VectorView::Magnetic;
        break;
    case '1': m_debugSpecies = DebugProjectileSpecies::Electron; break;
    case '2': m_debugSpecies = DebugProjectileSpecies::ArgonIon; break;
    case '3': m_debugSpecies = DebugProjectileSpecies::ArgonNeutral; break;
    case 'v':
        m_vectorView = static_cast<VectorView>((static_cast<int>(m_vectorView) + 1) %
            static_cast<int>(VectorView::Count));
        break;
    case 'b':
        m_scalarView = static_cast<ScalarView>((static_cast<int>(m_scalarView) + 1) %
            static_cast<int>(ScalarView::Count));
        break;
    case 'g':
        m_vectorRenderSettings.scale = static_cast<VectorGlyphScale>(
            (static_cast<int>(m_vectorRenderSettings.scale) + 1) % 3);
        break;
    case 'c':
        m_debugElectrodynamics.clear();
        refreshDiagnosticFields();
        m_debugNotice = "TEST PARTICLES AND THEIR FIELDS CLEARED";
        m_debugNoticeSeconds = 3.0f;
        break;
    }
    return true;
}

bool MultiPhysicsSimWorkspace::handlePointerInput(
    const WorkspacePointerEvent& input, WorkspaceServices& services) {
    if (!layer3Active()) return false;
    using Type = WorkspacePointerEvent::Type;
    using Button = WorkspacePointerEvent::Button;
    if (input.button == Button::Right) return false;
    if (input.type == Type::Button && input.button == Button::Left) {
        if (!input.pressed) {
            const bool captured = m_fireClickCaptured || m_freeLookDragging || m_testFireMode;
            m_fireClickCaptured = m_freeLookDragging = false;
            return captured;
        }
        if (m_testFireMode) {
            m_fireClickCaptured = true;
            m_freeLookDragging = false;
            glm::vec3 origin, direction;
            if (!services.camera || !services.camera->getCenterViewRay(origin, direction)) {
                m_debugNotice = "TEST FIRE: CAMERA RAY UNAVAILABLE";
            }
            else {
                const auto result = m_debugElectrodynamics.fire(m_debugSpecies, origin, direction, m_baseVoxelGrid);
                if (result == DebugElectrodynamics::FireResult::Fired) {
                    refreshDiagnosticFields();
                    m_debugNotice = std::string("TEST FIRE: ") + speciesName(m_debugSpecies) +
                        (m_paused ? " (PAUSED - SPACE TO ADVANCE)" : "");
                }
                else if (result == DebugElectrodynamics::FireResult::MissedDomain)
                    m_debugNotice = "TEST FIRE: CENTER RAY MISSED SIMULATION DOMAIN";
                else m_debugNotice = "TEST FIRE: INVALID CENTER RAY";
            }
            m_debugNoticeSeconds = 3.0f;
            return true;
        }
        if (m_layer3CameraView == Layer3CameraView::Free) {
            m_freeLookDragging = true;
            return true;
        }
    }
    if (input.type == Type::Motion) {
        if (m_fireClickCaptured || m_testFireMode) return true;
        if (m_freeLookDragging && services.camera) {
            services.camera->lookFree(static_cast<float>(input.dx), static_cast<float>(input.dy));
            return true;
        }
    }
    return false;
}

bool MultiPhysicsSimWorkspace::setFreeMovementKey(WorkspaceInputAction action, bool pressed) {
    int index = -1;
    switch (action) {
    case WorkspaceInputAction::Previous: index = 0; break;
    case WorkspaceInputAction::Next: index = 1; break;
    case WorkspaceInputAction::Decrease: index = 2; break;
    case WorkspaceInputAction::Increase: index = 3; break;
    default: return false;
    }
    m_freeMovementKeys[index] = pressed;
    return true;
}

bool MultiPhysicsSimWorkspace::handleInputRelease(
    const WorkspaceInputEvent& input, WorkspaceServices& services) {
    (void)services;
    return setFreeMovementKey(input.action, false);
}

void MultiPhysicsSimWorkspace::cancelInput(WorkspaceServices& services) {
    (void)services;
    m_fireClickCaptured = m_freeLookDragging = false;
    std::fill(std::begin(m_freeMovementKeys), std::end(m_freeMovementKeys), false);
}

void MultiPhysicsSimWorkspace::leaveRuntimeCamera(WorkspaceServices& services) {
    cancelInput(services);
    if (services.camera && m_layer3CameraView == Layer3CameraView::Free) services.camera->endFreeView();
    m_layer3CameraView = Layer3CameraView::Orbit;
}

WorkspaceMenuPresentation MultiPhysicsSimWorkspace::buildMenu() const {
    WorkspaceMenuPresentation menu;
    if (!layer3Active()) return menu;
    menu.items.push_back({"- MULTIPHYSICS FIELD / TEST -", 0, false});
    menu.items.push_back({std::string("* CAM VIEW [") +
        (m_layer3CameraView == Layer3CameraView::Free ? "FREE]" : "ORBIT]"), MenuCameraView, true});
    menu.items.push_back({std::string("* TEST FIRE [") + (m_testFireMode ? "ON]" : "OFF]"), MenuFireMode, true});
    menu.items.push_back({"* Clear test particles", MenuClearDebug, true});
    return menu;
}

bool MultiPhysicsSimWorkspace::handleMenuCommand(int command, WorkspaceServices& services) {
    if (!layer3Active()) return false;
    if (command == MenuCameraView && services.camera) {
        cancelInput(services);
        if (m_layer3CameraView == Layer3CameraView::Free) leaveRuntimeCamera(services);
        else {
            services.camera->beginFreeView();
            if (services.camera->freeViewActive()) m_layer3CameraView = Layer3CameraView::Free;
        }
        return true;
    }
    WorkspaceInputEvent input;
    input.action = WorkspaceInputAction::RawKey;
    if (command == MenuFireMode) input.rawKey = 'f';
    else if (command == MenuClearDebug) input.rawKey = 'c';
    else return false;
    return handleFieldDebugKey(input, services);
}

void MultiPhysicsSimWorkspace::renderOverlay(
    const WorkspaceFrameContext& frame, WorkspaceServices& services) {
    (void)services;
    if (frame.displayEnabled && layer3Active() && m_testFireMode)
        FieldDebugRenderer::drawCrosshair(frame.viewportWidth, frame.viewportHeight);
}

void MultiPhysicsSimWorkspace::appendFieldDebugStatus(WorkspaceRuntimeStatus& status) const {
    const char* scale = m_vectorRenderSettings.scale == VectorGlyphScale::Normalized ? "DIRECTION" :
        m_vectorRenderSettings.scale == VectorGlyphScale::RelativeMagnitude ? "RELATIVE" : "LOG";
    const auto* vector = selectedVectorField();
    const auto* scalar = selectedScalarField();
    status.detailLines.push_back(std::string("FIELD VIEW: ") + vectorViewName() + " | SCALE: " + scale +
        (vector ? " | MAX: " + magnitudeText(vector->maxMagnitude()) : ""));
    status.detailLines.push_back(std::string("SCALAR: ") + scalarViewName() +
        (scalar ? " | MAX ABS: " + magnitudeText(scalar->maxMagnitude()) : "") +
        (m_scalarView == ScalarView::ChargeDensity ? " | BLUE - / ORANGE +" : ""));
    status.detailLines.push_back(std::string("TEST FIRE: ") + (m_testFireMode ? "ON" : "OFF") +
        " | " + speciesName(m_debugSpecies) + " | FIRED: " + std::to_string(m_debugElectrodynamics.firedCount()) +
        " | ACTIVE DEBUG: " + std::to_string(m_debugElectrodynamics.projectiles().size()) + "/128");
    std::ostringstream speed;
    speed << std::fixed << std::setprecision(2) << m_debugElectrodynamics.speedMps();
    status.detailLines.push_back(std::string("QUASI-STATIC DEBUG: FIRED SOURCES ONLY | SPEED: ") + speed.str() +
        " m/s | CAM: " + (m_layer3CameraView == Layer3CameraView::Free ? "FREE (WASD, F OFF: DRAG TO LOOK)" : "ORBIT"));
    status.detailLines.push_back("F: Fire mode | 1: Electron  2: Ar+  3: Ar | LEFT CLICK: Center ray");
    status.detailLines.push_back("V: Vector view | B: Scalar view | G: Arrow scale | C: Clear tests");
    if (m_debugNoticeSeconds > 0.0f) status.detailLines.push_back(m_debugNotice);
}
