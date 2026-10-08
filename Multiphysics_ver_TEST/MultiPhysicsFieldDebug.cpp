#include "atomicParticlesSimWorkspace.h"
#include "CameraEM.h"
#include "kernel.h"
#include "rendererEM_Euclid.h"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <sstream>

namespace {
    // Display palette and stable SI display reference; neither changes field values.
    const glm::vec4 kPositiveElectricColor(1.0f, 0.12f, 0.08f, 0.9f);
    const glm::vec4 kNegativeElectricColor(0.12f, 0.45f, 1.0f, 0.9f);
    const glm::vec4 kDipoleElectricColor(0.72f, 0.25f, 1.0f, 0.9f);
    const glm::vec4 kPositiveMagneticColor(0.1f, 1.0f, 0.15f, 0.9f);
    const glm::vec4 kNegativeMagneticColor(1.0f, 0.9f, 0.08f, 0.9f);
    // Approximately ten elementary-charge fields at 1 um; fixed across frames
    // and presets so a moving source changes arrow length at a fixed voxel.
    constexpr double kElectricGlyphReferenceVm = 1.44e4; // ten elementary charges at 1 um

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

bool AtomicParticlesSimWorkspace::layer3Active() const {
    return m_active && m_arbiter &&
        m_arbiter->getApplicationLayer() == TheArbiter::ApplicationLayer::ACTIVE_WORKSPACE &&
        m_arbiter->getActiveWorkspace() == TheArbiter::WorkspaceId::ATOMIC_PARTICLES;
}

void AtomicParticlesSimWorkspace::initializeFields() {
    m_electronDensity.initialize(m_fieldVoxelGrid);
    m_electronTemperature.initialize(m_fieldVoxelGrid);
    m_chargeDensity.initialize(m_fieldVoxelGrid);
    m_electricField.initialize(m_fieldVoxelGrid);
    m_magneticField.initialize(m_fieldVoxelGrid);
    m_currentDensity.initialize(m_fieldVoxelGrid);
    m_curlMagneticField.initialize(m_fieldVoxelGrid);
    m_fieldVisualization.clear();
    m_electricGlyphColors.clear();
    m_magneticGlyphColors.clear();
    const auto& g = m_fieldVoxelGrid;
    const FieldGridParams grid{make_uint3(g.dimensions.x,g.dimensions.y,g.dimensions.z),
        make_float3(g.origin.x,g.origin.y,g.origin.z),
        make_float3(g.voxelEdgeM,g.voxelEdgeM,g.voxelEdgeM),g.voxelCount(),g.metersPerWorldUnit};
    bool ready = false;
    if (m_fieldSystem) ready = m_fieldSystem->setFieldGrid(grid);
    else {
        m_fieldSystem = std::make_unique<FieldSystem>(grid,true);
        ready = m_fieldSystem->initialized();
    }
    if (ready) {
        auto solver=m_fieldSystem->getSolverParams();
        solver.coulombSofteningM=static_cast<float>(0.25*g.voxelEdgeM*g.metersPerWorldUnit);
        ready=m_fieldSystem->setSolverParams(solver);
    }
    m_fieldMirrorReady = ready;
    m_fieldGlyphPalette.resize(grid.cellCount);
    if (!ready) {
        m_fieldSystem.reset(); // No stale geometry after failed replacement.
        m_fieldBackend = FieldBackend::CPU;
        m_debugNotice = "CUDA field resources unavailable; CPU field display active";
        m_debugNoticeSeconds = 5.0f;
    }
}

void AtomicParticlesSimWorkspace::clearFieldDebug() {
    m_debugElectrodynamics.clear();
    m_electronDensity.clear();
    m_electronTemperature.clear();
    m_chargeDensity.clear();
    m_electricField.clear();
    m_magneticField.clear();
    m_currentDensity.clear();
    m_curlMagneticField.clear();
    m_fieldVisualization.clear();
    m_electricGlyphColors.clear();
    m_magneticGlyphColors.clear();
    m_testFireMode = m_fireClickCaptured = m_freeLookDragging = false;
    std::fill(std::begin(m_freeMovementKeys), std::end(m_freeMovementKeys), false);
    m_vectorView = VectorView::Off;
    m_scalarView = ScalarView::Off;
    m_debugSpecies = DebugProjectileSpecies::Electron;
    m_debugNotice.clear();
    m_debugNoticeSeconds = 0.0f;
    m_fieldMirrorReady = m_fieldSystem && m_fieldSystem->clear();
}

void AtomicParticlesSimWorkspace::refreshDiagnosticFields() {
    if (m_runtimeEnabled && m_activeMarkerCount > 0 && m_fieldSystem && !m_testFireMode) {
        // Production fields must never be overwritten by the independent fired-source preview.
        m_fieldMirrorReady=m_fieldSystem->downloadFields();
        if (!m_fieldMirrorReady) { m_debugNotice="Field readback failed"; m_debugNoticeSeconds=5; return; }
        const auto& rho=m_fieldSystem->getScalarHost(FieldSystem::CHARGE_DENSITY);
        const auto& e=m_fieldSystem->getVectorHost(FieldSystem::ELECTRIC_FIELD);
        const auto& b=m_fieldSystem->getVectorHost(FieldSystem::MAGNETIC_FIELD);
        const auto& j=m_fieldSystem->getVectorHost(FieldSystem::CURRENT_DENSITY);
        for (unsigned i=0;i<rho.size();++i) {
            m_chargeDensity.set(i,rho[i]);
            m_electricField.set(i,glm::vec3(e[i].x,e[i].y,e[i].z));
            m_magneticField.set(i,glm::vec3(b[i].x,b[i].y,b[i].z));
            m_currentDensity.set(i,glm::vec3(j[i].x,j[i].y,j[i].z));
        }
        computeCurl(m_magneticField,m_curlMagneticField);
        m_electricGlyphColors.clear(); m_magneticGlyphColors.clear();
        m_fieldSystem->setElectricGlyphColors({});
        return;
    }
    // Only explicitly fired debug projectiles source these diagnostics.
    // The production CUDA population is not a self-consistent PIC plasma.
    m_debugElectrodynamics.populateFields(m_fieldVoxelGrid, m_electricField,
        m_magneticField, m_currentDensity, m_chargeDensity, &m_fieldVisualization);
    computeCurl(m_magneticField, m_curlMagneticField);
    m_electricGlyphColors.resize(m_fieldVisualization.size());
    m_magneticGlyphColors.resize(m_fieldVisualization.size());
    for (std::size_t id = 0; id < m_fieldVisualization.size(); id++) {
        const auto& sample = m_fieldVisualization[id];
        switch (sample.electricClass) {
        case ElectricGlyphClass::PositiveSource: m_electricGlyphColors[id] = kPositiveElectricColor; break;
        case ElectricGlyphClass::NegativeSource: m_electricGlyphColors[id] = kNegativeElectricColor; break;
        case ElectricGlyphClass::DipoleBridge: m_electricGlyphColors[id] = kDipoleElectricColor; break;
        }
        // Draw the physical net B; color indicates its locally dominant source polarity.
        m_magneticGlyphColors[id] = vectorMagnitude(sample.magneticPositive) >= vectorMagnitude(sample.magneticNegative)
            ? kPositiveMagneticColor : kNegativeMagneticColor;
    }
    m_fieldMirrorReady = false;
    if (m_fieldBackend == FieldBackend::CUDA && !mirrorDiagnosticFields()) {
        m_fieldBackend = FieldBackend::CPU;
        m_debugNotice = "CUDA field upload failed; CPU field display active";
        m_debugNoticeSeconds = 5.0f;
    }
}

bool AtomicParticlesSimWorkspace::mirrorDiagnosticFields() {
    if (m_runtimeEnabled && m_activeMarkerCount > 0) {
        // Preview upload is scoped to renderFieldDebug; physics buffers stay owned by the solver.
        m_fieldMirrorReady=m_fieldSystem && m_fieldSystem->initialized();
        return m_fieldMirrorReady;
    }
    if (!m_fieldSystem || !m_fieldSystem->initialized() ||
        m_fieldSystem->getCellCount() != m_chargeDensity.size()) return false;
    // Explicit transition path: fired CPU diagnostics -> GPU display fields.
    // No production particle deposition or field forces are implied here.
    auto& rho = m_fieldSystem->getScalarHost(FieldSystem::CHARGE_DENSITY);
    auto& ne = m_fieldSystem->getScalarHost(FieldSystem::ELECTRON_DENSITY);
    auto& te = m_fieldSystem->getScalarHost(FieldSystem::ELECTRON_TEMPERATURE);
    auto& e = m_fieldSystem->getVectorHost(FieldSystem::ELECTRIC_FIELD);
    auto& b = m_fieldSystem->getVectorHost(FieldSystem::MAGNETIC_FIELD);
    auto& j = m_fieldSystem->getVectorHost(FieldSystem::CURRENT_DENSITY);
    for (unsigned i=0; i<m_fieldSystem->getCellCount(); ++i) {
        rho[i]=static_cast<float>(m_chargeDensity.get(i));
        ne[i]=static_cast<float>(m_electronDensity.get(i));
        te[i]=static_cast<float>(m_electronTemperature.get(i));
        const auto ev=m_electricField.get(i), bv=m_magneticField.get(i), jv=m_currentDensity.get(i);
        e[i]=make_float4(ev.x,ev.y,ev.z,0);
        b[i]=make_float4(bv.x,bv.y,bv.z,0);
        j[i]=make_float4(jv.x,jv.y,jv.z,0);
        const auto color=m_electricGlyphColors.size()==e.size() ? m_electricGlyphColors[i] : kPositiveElectricColor;
        m_fieldGlyphPalette[i]=make_float4(color.r,color.g,color.b,color.a);
    }
    m_fieldMirrorReady = m_fieldSystem->uploadFields() &&
        m_fieldSystem->setElectricGlyphColors(m_fieldGlyphPalette);
    return m_fieldMirrorReady;
}

void AtomicParticlesSimWorkspace::updateFieldDebug(
    const WorkspaceFrameContext& frame, WorkspaceServices& services) {
    m_debugNoticeSeconds = (std::max)(0.0f, m_debugNoticeSeconds - frame.deltaTime);
    if (!m_subLayers.panelOpen() && m_layer3CameraView == Layer3CameraView::Free && services.camera) {
        services.camera->moveFree(
            static_cast<float>(m_freeMovementKeys[0]) - static_cast<float>(m_freeMovementKeys[1]),
            static_cast<float>(m_freeMovementKeys[3]) - static_cast<float>(m_freeMovementKeys[2]),
            frame.deltaTime);
    }
    if (!m_runtimeEnabled || m_paused || frame.deltaTime <= 0.0f ||
        m_debugElectrodynamics.projectiles().empty()) return;
    m_debugElectrodynamics.update(double(frame.deltaTime)*kPhysicalSecondsPerWallSecond, m_fieldVoxelGrid);
    refreshDiagnosticFields();
}

const VectorField3D* AtomicParticlesSimWorkspace::selectedVectorField() const {
    switch (m_vectorView) {
    case VectorView::Electric: return &m_electricField;
    case VectorView::Magnetic: return &m_magneticField;
    case VectorView::Current: return &m_currentDensity;
    case VectorView::CurlB: return &m_curlMagneticField;
    default: return nullptr;
    }
}

const ScalarField3D* AtomicParticlesSimWorkspace::selectedScalarField() const {
    switch (m_scalarView) {
    case ScalarView::ElectronDensity: return &m_electronDensity;
    case ScalarView::ElectronTemperature: return &m_electronTemperature;
    case ScalarView::ChargeDensity: return &m_chargeDensity;
    default: return nullptr;
    }
}

const char* AtomicParticlesSimWorkspace::vectorViewName() const {
    switch (m_vectorView) {
    case VectorView::Electric: return "ELECTRIC_FIELD [V/m]";
    case VectorView::Magnetic: return "MAGNETIC_FIELD [T]";
    case VectorView::Current: return "CURRENT_DENSITY [A/m^2]";
    case VectorView::CurlB: return "CURL_B [T/m]";
    default: return "OFF";
    }
}

const char* AtomicParticlesSimWorkspace::scalarViewName() const {
    switch (m_scalarView) {
    case ScalarView::ElectronDensity: return "ELECTRON_DENSITY [m^-3] (ZERO PLACEHOLDER)";
    case ScalarView::ElectronTemperature: return "ELECTRON_TEMPERATURE [eV] (ZERO PLACEHOLDER)";
    case ScalarView::ChargeDensity: return "CHARGE_DENSITY [C/m^3]";
    default: return "OFF";
    }
}

void AtomicParticlesSimWorkspace::renderFieldDebug(WorkspaceServices& services) {
    if (!services.renderer) return;
    FieldDebugRenderer::drawProjectiles(*services.renderer, m_debugElectrodynamics.projectiles());
    FieldDebugRenderer::drawProjectileVelocities(m_debugElectrodynamics.projectiles(), m_fieldVoxelGrid.voxelEdgeM);
    bool gpuRendered = false;
    bool diagnosticSnapshot = false;
    // Test-fire is an explicitly separate visualization. Temporarily stage its
    // E/scalar into the existing render buffers, then restore the physical fields.
    // Existing FieldSystem host mirrors hold the snapshot; no second field owner.
    if (m_activeMarkerCount && m_testFireMode && m_fieldBackend == FieldBackend::CUDA && m_fieldSystem) {
        diagnosticSnapshot=m_fieldSystem->downloadFields();
        if (diagnosticSnapshot) {
            std::vector<float4> electric(m_electricField.size());
            std::vector<float> scalar(electric.size());
            const auto* selected=selectedScalarField();
            for (unsigned i=0;i<electric.size();++i) {
                const auto value=m_electricField.get(i);
                electric[i]=make_float4(value.x,value.y,value.z,0);
                scalar[i]=selected ? static_cast<float>(selected->get(i)) : 0;
                const auto color=m_electricGlyphColors.size()==electric.size() ? m_electricGlyphColors[i] : kPositiveElectricColor;
                m_fieldGlyphPalette[i]=make_float4(color.r,color.g,color.b,color.a);
            }
            const auto channel=m_scalarView==ScalarView::ElectronDensity ? FieldSystem::ELECTRON_DENSITY :
                m_scalarView==ScalarView::ElectronTemperature ? FieldSystem::ELECTRON_TEMPERATURE : FieldSystem::CHARGE_DENSITY;
            m_fieldMirrorReady=copyFieldToDevice(m_fieldSystem->getVectorDevicePtr(FieldSystem::ELECTRIC_FIELD),electric.data(),electric.size()*sizeof(float4)) &&
                copyFieldToDevice(m_fieldSystem->getScalarDevicePtr(channel),scalar.data(),scalar.size()*sizeof(float)) &&
                m_fieldSystem->setElectricGlyphColors(m_fieldGlyphPalette);
        }
        if (!diagnosticSnapshot || !m_fieldMirrorReady) {
            m_fieldBackend=FieldBackend::CPU;
            m_debugNotice="Diagnostic upload failed; CPU preview active"; m_debugNoticeSeconds=5;
        }
    }
    const bool gpuSelected = m_vectorView == VectorView::Electric || selectedScalarField();
    if (m_fieldBackend == FieldBackend::CUDA && gpuSelected &&
        (!m_fieldSystem || (!m_fieldMirrorReady && !mirrorDiagnosticFields()))) {
        m_fieldBackend = FieldBackend::CPU;
        m_debugNotice = "CUDA field upload unavailable; CPU display active";
        m_debugNoticeSeconds = 5.0f;
    }
    if (m_fieldBackend == FieldBackend::CUDA && gpuSelected) {
        FieldRenderParams settings;
        settings.vectorScale = static_cast<unsigned>(m_vectorRenderSettings.scale);
        settings.lengthInCells = m_vectorRenderSettings.lengthInVoxels;
        settings.vectorReference = kElectricGlyphReferenceVm;
        settings.logStrength = m_vectorRenderSettings.logStrength;
        const auto* scalar = selectedScalarField();
        settings.scalarReference = scalar && scalar->maxMagnitude()>0 ? scalar->maxMagnitude() : 1.0;
        const auto channel = m_scalarView == ScalarView::ElectronDensity ? FieldSystem::ELECTRON_DENSITY :
            m_scalarView == ScalarView::ElectronTemperature ? FieldSystem::ELECTRON_TEMPERATURE : FieldSystem::CHARGE_DENSITY;
        gpuRendered = m_fieldSystem->buildRenderBuffers(channel,settings);
        if (gpuRendered) {
            services.renderer->setFieldSystem(m_fieldSystem.get());
            if (m_vectorView == VectorView::Electric) services.renderer->displayElectricField();
            if (scalar) services.renderer->displayScalarField();
            // Borrow only for this draw, so resize and exit cannot leave a stale pointer.
            services.renderer->setFieldSystem(nullptr);
        } else {
            m_fieldBackend = FieldBackend::CPU;
            m_debugNotice = "CUDA field drawing unavailable; CPU display active";
            m_debugNoticeSeconds = 5.0f;
        }
    }
    if (const auto* field = (gpuRendered && m_vectorView == VectorView::Electric) ? nullptr : selectedVectorField()) {
        auto settings = m_vectorRenderSettings;
        if (m_vectorView == VectorView::Electric) {
            settings.referenceMagnitude = kElectricGlyphReferenceVm;
            FieldDebugRenderer::drawVector(*field, m_electricGlyphColors, settings);
        }
        else if (m_vectorView == VectorView::Magnetic)
            FieldDebugRenderer::drawVector(*field, m_magneticGlyphColors, settings);
        else FieldDebugRenderer::drawVector(*field, settings);
    }
    if (!gpuRendered) if (const auto* field = selectedScalarField()) FieldDebugRenderer::drawScalar(*field);
    if (diagnosticSnapshot) {
        m_fieldMirrorReady=m_fieldSystem->uploadFields();
        m_fieldSystem->setElectricGlyphColors({});
        if (!m_fieldMirrorReady) { m_paused=true; m_debugNotice="Physics paused: field snapshot restore failed"; m_debugNoticeSeconds=5; }
    }
}

bool AtomicParticlesSimWorkspace::handleFieldDebugKey(
    const WorkspaceInputEvent& input, WorkspaceServices& services) {
    const unsigned char key = static_cast<unsigned char>(std::tolower(input.rawKey));
    if (key != 'f' && key != 'v' && key != 'b' && key != 'c' && key != 'g' &&
        key != '1' && key != '2' && key != '3' && key != 'h') return false;
    if (input.repeated) return true;
    switch (key) {
    case 'h':
        if (m_fieldBackend == FieldBackend::CUDA) m_fieldBackend = FieldBackend::CPU;
        else if (mirrorDiagnosticFields()) m_fieldBackend = FieldBackend::CUDA;
        else {
            m_debugNotice = "CUDA field resources unavailable; CPU display active";
            m_debugNoticeSeconds = 5.0f;
        }
        break;
    case 'f':
        cancelInput(services);
        m_testFireMode = !m_testFireMode;
        if (m_testFireMode && m_vectorView == VectorView::Off) m_vectorView = VectorView::Electric;
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
    refreshDiagnosticFields();
    return true;
}

bool AtomicParticlesSimWorkspace::handlePointerInput(
    const WorkspacePointerEvent& input, WorkspaceServices& services) {
    if (!layer3Active()) return false;
    using Type = WorkspacePointerEvent::Type;
    using Button = WorkspacePointerEvent::Button;
    if (input.button == Button::Right) return false;
    if (m_subLayers.panelOpen()) return true;
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
                const auto result = m_debugElectrodynamics.fire(m_debugSpecies, origin, direction, m_fieldVoxelGrid);
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

bool AtomicParticlesSimWorkspace::setFreeMovementKey(WorkspaceInputAction action, bool pressed) {
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

bool AtomicParticlesSimWorkspace::handleInputRelease(
    const WorkspaceInputEvent& input, WorkspaceServices& services) {
    (void)services;
    return setFreeMovementKey(input.action, false);
}

void AtomicParticlesSimWorkspace::cancelInput(WorkspaceServices& services) {
    (void)services;
    m_fireClickCaptured = m_freeLookDragging = false;
    std::fill(std::begin(m_freeMovementKeys), std::end(m_freeMovementKeys), false);
}

void AtomicParticlesSimWorkspace::leaveRuntimeCamera(WorkspaceServices& services) {
    cancelInput(services);
    if (services.camera && m_layer3CameraView == Layer3CameraView::Free) services.camera->endFreeView();
    m_layer3CameraView = Layer3CameraView::Orbit;
}

WorkspaceMenuPresentation AtomicParticlesSimWorkspace::buildMenu() const {
    WorkspaceMenuPresentation menu;
    if (!layer3Active()) return menu;
    menu.items.push_back({"- ATOMIC PARTICLES FIELD / TEST -", 0, false});
    menu.items.push_back({std::string("* CAM VIEW [") +
        (m_layer3CameraView == Layer3CameraView::Free ? "FREE]" : "ORBIT]"), MenuCameraView, true});
    menu.items.push_back({std::string("* TEST FIRE [") + (m_testFireMode ? "ON]" : "OFF]"), MenuFireMode, true});
    menu.items.push_back({"* Clear test particles", MenuClearDebug, true});
    menu.items.push_back({std::string("* FIELD DISPLAY [") +
        (m_fieldBackend == FieldBackend::CUDA ? "CUDA]" : "CPU]"), MenuFieldBackend, true});
    return menu;
}

bool AtomicParticlesSimWorkspace::handleMenuCommand(int command, WorkspaceServices& services) {
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
    else if (command == MenuFieldBackend) input.rawKey = 'h';
    else return false;
    return handleFieldDebugKey(input, services);
}

void AtomicParticlesSimWorkspace::renderOverlay(
    const WorkspaceFrameContext& frame, WorkspaceServices& services) {
    (void)services;
    if (frame.displayEnabled && layer3Active() && m_testFireMode)
        FieldDebugRenderer::drawCrosshair(frame.viewportWidth, frame.viewportHeight);
}

void AtomicParticlesSimWorkspace::appendFieldDebugStatus(WorkspaceRuntimeStatus& status) const {
    std::ostringstream physicalTime;
    physicalTime << std::scientific << std::setprecision(3) << m_elapsedSimulationTime;
    status.detailLines.push_back("PHYSICS: 1 WORLD UNIT = 1 um | GRID ELECTROSTATICS | t = " + physicalTime.str() + " s");
    status.detailLines.push_back("TIME: 10 ps / wall second | MAX STEP: 0.1 ps | E/B: SI; acceleration: m/s^2");
    status.detailLines.push_back(m_testFireMode || !m_activeMarkerCount ? "FIELD SOURCES: FIRED DIAGNOSTIC PREVIEW (UNCOUPLED)" : "FIELD SOURCES: CUDA POPULATION | GRID ELECTROSTATICS");
    const char* scale = m_vectorRenderSettings.scale == VectorGlyphScale::Normalized ? "DIRECTION" :
        m_vectorRenderSettings.scale == VectorGlyphScale::RelativeMagnitude ? "RELATIVE" : "LOG";
    const auto* vector = selectedVectorField();
    const auto* scalar = selectedScalarField();
    status.detailLines.push_back(std::string("FIELD VIEW: ") + vectorViewName() + " | SCALE: " + scale +
        (vector ? " | MAX: " + magnitudeText(vector->maxMagnitude()) : ""));
    status.detailLines.push_back("VELOCITY: ORANGE | E: RED + / BLUE - / PURPLE BRIDGE | B: GREEN + / YELLOW -");
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
    status.detailLines.push_back(std::string("H: FIELD DISPLAY: ") +
        (m_fieldBackend == FieldBackend::CUDA ? "CUDA E/SCALAR" : "CPU DISPLAY OF SI FIELDS"));
    if (m_fieldSystem && !m_fieldSystem->waveSpatiallyResolved())
        status.detailLines.push_back("ANALYTIC WAVE UNDER-RESOLVED: fewer than 10 cells per wavelength");
    if (m_debugNoticeSeconds > 0.0f) status.detailLines.push_back(m_debugNotice);
}
