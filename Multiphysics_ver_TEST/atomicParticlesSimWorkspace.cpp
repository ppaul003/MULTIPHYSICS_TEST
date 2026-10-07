#include "atomicParticlesSimWorkspace.h"
#include "SimulationPreset.h"

#include "rendererEM_Euclid.h"

#include <string>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

using namespace std;
using namespace glm;

namespace {

    WorkspacePanelRow makeRow(
        const string& label,
        const string& value,
        bool selected) {

        WorkspacePanelRow row;
        row.label = label;
        row.value = value;
        row.selectable = true;
        row.selected = selected;

        return row;
    }

} // namespace

bool AtomicParticlesSimWorkspace::initialize(WorkspaceServices& services) {
    if (m_initialized) return true;

    if (!services.renderer ||
        !services.arbiter)
        return false;

    m_arbiter = services.arbiter;


    m_radii.assign(kParticleCapacity, 0.0f);

    m_colors.assign(
        kParticleCapacity,
        make_float4(1.0f, 1.0f, 1.0f, 1.0f)
    );

    const uint3 collisionGrid =
        make_uint3(kGridSize, kGridSize, kGridSize);

    m_particleSystem =
        make_unique<ParticleSystem>(kParticleCapacity, collisionGrid, true);

    syncSimulationDomain(services);
    if (!m_particleSystem->setActiveParticleCount(0))
        return false;

    m_initialized = true;
    return true;
}

void AtomicParticlesSimWorkspace::syncSimulationDomain(WorkspaceServices& services) {
    if (!services.renderer || !m_particleSystem) return;
    const float boxSize = static_cast<float>(services.renderer->getSimBoxSize());
    const auto* preset = findSimulationPreset(services.renderer->getSimBoxSize());
    if (!preset || !preset->enabled || services.renderer->getGridDimSize() != preset->collisionGridDim) return;
    const auto currentGrid = m_particleSystem->getGridSize();
    const uint dimension = static_cast<uint>(preset->collisionGridDim);
    if (boxSize == m_simulationBoxSizeM && currentGrid.x == dimension &&
        currentGrid.y == dimension && currentGrid.z == dimension &&
        m_fieldVoxelGrid.dimensions == ivec3(preset->fieldGridDim)) return;
    m_particleSystem->setSimulationDomain(boxSize, make_uint3(dimension, dimension, dimension));
    clearRuntime();
    m_simulationBoxSizeM = boxSize;
    m_baseVoxelGrid.dimensions = ivec3(8);
    m_baseVoxelGrid.origin = vec3(-boxSize * 0.5f);
    m_baseVoxelGrid.voxelEdgeM = boxSize / 8.0f;
    m_fieldVoxelGrid.dimensions = ivec3(preset->fieldGridDim);
    m_fieldVoxelGrid.origin = vec3(-boxSize * 0.5f);
    m_fieldVoxelGrid.voxelEdgeM = boxSize / static_cast<float>(preset->fieldGridDim);
    // Reinitialize fields against the independent sampling grid after a commit.
    initializeFields();
}

SimulationDomainState AtomicParticlesSimWorkspace::simulationDomainState() const {
    SimulationDomainState state;
    state.physicalGrid = m_baseVoxelGrid;
    state.fieldGrid = m_fieldVoxelGrid;
    if (!m_particleSystem) return state;
    const auto dim = m_particleSystem->getGridSize();
    const auto origin = m_particleSystem->getWorldOrigin();
    const auto cell = m_particleSystem->getCellSize();
    state.collisionDimensions = ivec3(dim.x, dim.y, dim.z);
    state.collisionOrigin = vec3(origin.x, origin.y, origin.z);
    state.collisionCellSize = vec3(cell.x, cell.y, cell.z);
    state.collisionRadius = m_particleSystem->getParticleRadius();
    state.collisionCellCount = m_particleSystem->getNumGridCells();
    state.fieldCellCount = m_chargeDensity.size();
    const auto valid = [&](const auto& field) {
        return field.initialized() && &field.grid() == &m_fieldVoxelGrid &&
            field.size() == m_fieldVoxelGrid.voxelCount();
    };
    state.fieldGeometryValid = valid(m_electronDensity) && valid(m_electronTemperature) &&
        valid(m_chargeDensity) && valid(m_electricField) && valid(m_magneticField) &&
        valid(m_currentDensity) && valid(m_curlMagneticField);
    return state;
}

void AtomicParticlesSimWorkspace::enter(WorkspaceServices& services) {
    syncSimulationDomain(services);
    if (!m_arbiter) m_arbiter = services.arbiter;
    if (!m_initialized) return;

    clearFieldDebug();
    m_active = true;
    m_paused = true;
    m_runtimeEnabled = false;
    m_activeMarkerCount = 0;
    m_elapsedSimulationTime = 0.0f;

    refreshLayer1Status();
}

void AtomicParticlesSimWorkspace::exit(WorkspaceServices& services) {
    leaveRuntimeCamera(services);
    clearFieldDebug();
    m_active = false;
    m_paused = true;
    m_runtimeEnabled = false;
    m_activeMarkerCount = 0;
}

void AtomicParticlesSimWorkspace::update(
    const WorkspaceFrameContext& frame,
    WorkspaceServices& services) {

    (void)services;

    if (!m_active ||
        !m_arbiter ||
        !m_particleSystem) return;

    if (m_arbiter->getApplicationLayer() != TheArbiter::ApplicationLayer::ACTIVE_WORKSPACE)
        return;

    updateFieldDebug(frame, services);
    if (!m_runtimeEnabled || m_paused) return;

    m_particleSystem->update(frame.deltaTime);

    m_elapsedSimulationTime += frame.deltaTime;
}

void AtomicParticlesSimWorkspace::render(
    const WorkspaceFrameContext& frame,
    WorkspaceServices& services) {

    if (!frame.displayEnabled ||
        !services.renderer ||
        !services.arbiter) {
        return;
    }

    switch (services.arbiter->getApplicationLayer()) {

    case TheArbiter::ApplicationLayer::DOMAIN_SELECTION:
        renderConfiguredGrid(services, m_draftConfig.gridLayout);
        return;

    case TheArbiter::ApplicationLayer::WORKSPACE_CONFIGURATION:
        renderConfiguredGrid(services, m_draftConfig.gridLayout);
        if (m_runtimeEnabled) {
            renderActivePlasmaMarkers(services);
            renderFieldDebug(services);
        }
        if (m_layer2Selection == Layer2Row::VoxelSpawn) {
            renderSelectedSpawnRegion(services);
        }
        return;

    case TheArbiter::ApplicationLayer::ACTIVE_WORKSPACE:
        renderConfiguredGrid(services, m_runtimeConfig.gridLayout);
        renderActivePlasmaMarkers(services);
        renderFieldDebug(services);
        return;

    case TheArbiter::ApplicationLayer::GLOBAL_SHELL:
    default:
        return;

    }
}

void AtomicParticlesSimWorkspace::renderActivePlasmaMarkers(WorkspaceServices& services) {

    if (!services.renderer ||
        !m_particleSystem ||
        m_activeMarkerCount == 0) return;

    EuclidRenderer& renderer = *services.renderer;
    renderer.setParticleSystem(m_particleSystem.get());

    renderer.setVertexBuffer(
        m_particleSystem->getCurrentReadBuffer(),
        static_cast<int>(m_activeMarkerCount)
    );

    renderer.setColorBuffer(m_particleSystem->getColorBuffer());
    renderer.setRadius(m_radii.data(), static_cast<int>(m_activeMarkerCount));

    const int neutralStart = 0;
    const int ionStart = static_cast<int>(m_runtimeConfig.neutralCount);
    const int electronStart = ionStart + static_cast<int>(m_runtimeConfig.ionCount);

    // BASE SPHERES

    // Neutral population
    renderer.displayParticleRange(
        neutralStart,
        static_cast<int>(m_runtimeConfig.neutralCount),
        false
    );

    // Ion base sphere
    renderer.displayParticleRange(
        ionStart,
        static_cast<int>(m_runtimeConfig.ionCount),
        false
    );

    // Electron base sphere
    renderer.displayParticleRange(
        electronStart,
        static_cast<int>(m_runtimeConfig.electronCount),
        false
    );

    // Visual brightness multiplier; does not affect particle physics or size.
    constexpr float kIonEmissionIntensity = 2.5f;
    constexpr float kElectronEmissionIntensity = 2.5f;

    // EMISSIVE OVERLAY
    renderer.displayParticleRange(
        ionStart,
        static_cast<int>(m_runtimeConfig.ionCount),
        true,
        kIonEmissionIntensity
    );

    renderer.displayParticleRange(
        electronStart,
        static_cast<int>(m_runtimeConfig.electronCount),
        true,
        kElectronEmissionIntensity
    );
}

bool AtomicParticlesSimWorkspace::resolveRuntimeConfig(RuntimeConfig& resolved) const {
    if (m_draftConfig.gridLayout == GridLayout::Dynamic)
        return false;

    if (m_draftConfig.simulationBoundary != SimulationBoundary::Closed)
        return false;

    if (m_draftConfig.spawnSelectionIndex >=
        m_spawnDensityGrid.selectionCount(m_baseVoxelGrid))
        return false;

    const unsigned int ions = ionCount();
    const unsigned int neutrals = neutralCount();
    const unsigned int electrons = electronCount();

    const unsigned long long markerCount =
        static_cast<unsigned long long>(neutrals) +
        static_cast<unsigned long long>(ions) +
        static_cast<unsigned long long>(electrons);

    if (markerCount > kParticleCapacity) return false;

    resolved.particleSpecies = m_draftConfig.particleSpecies;
    resolved.totalGasCount = m_draftConfig.totalGasDensity;

    resolved.neutralCount = neutrals;
    resolved.ionCount = ions;
    resolved.electronCount = electrons;

    resolved.activeMarkerCount =
        static_cast<unsigned int>(markerCount);

    resolved.ionizationFraction = m_draftConfig.ionizationFraction;
    resolved.electronTemperatureEv = m_draftConfig.electronTemperature;
    resolved.ionTemperatureEv = m_draftConfig.ionTemperature;
    resolved.neutralTemperatureK = m_draftConfig.neutralTemperature;
    resolved.selectedSpawnSelectionIndex = m_draftConfig.spawnSelectionIndex;

    resolved.speciesRadius = selectedSpeciesRenderRadius();
    resolved.placementRadius = std::max(resolved.speciesRadius, kElectronRadius);
    resolved.gridLayout = m_draftConfig.gridLayout;

    return true;
}

bool AtomicParticlesSimWorkspace::configureRuntimeVisuals() {
    if (!m_particleSystem) return false;

    const unsigned int neutralEnd = m_runtimeConfig.neutralCount;
    const unsigned int ionEnd = neutralEnd + m_runtimeConfig.ionCount;
    const unsigned int markerEnd = ionEnd + m_runtimeConfig.electronCount;
    // An empty base population still supports the separate diagnostic launcher.
    if (markerEnd == 0) { m_radii.clear(); m_colors.clear(); return true; }

    m_radii.resize(markerEnd);
    m_colors.resize(markerEnd);

    const float speciesRadius = m_runtimeConfig.speciesRadius;

    // Neutral
    for (unsigned int i = 0; i < neutralEnd; i++) {

        m_radii[i] = speciesRadius;

        // Neutral species: standard shaded red.
        m_colors[i] = make_float4(1.0f, 0.05f, 0.0f, 1.0f);
    }

    // Ion
    for (unsigned int i = neutralEnd; i < ionEnd; i++) {

        m_radii[i] = speciesRadius;

        // Ion species: emissive green.
        m_colors[i] = make_float4(0.05f, 1.00f, 0.02f, 1.0f);
    }

    // Electron
    for (unsigned int i = ionEnd; i < markerEnd; i++) {

        m_radii[i] = kElectronRadius;

        m_colors[i] = make_float4(0.05f, 0.25f, 1.00f, 1.0f);
    }

    if (!m_particleSystem->setActiveRadii(m_radii.data(), markerEnd))
        return false;

    if (!m_particleSystem->setActiveColors(m_colors.data(), markerEnd))
        return false;

    return true;
}

bool AtomicParticlesSimWorkspace::runtimeMatchesDraft() const {
    // Compare values, not edit events: changing a value back should resume.
    const auto sameFloat = [](float a, float b) {
        return std::fabs(a - b) <= 1.0e-5f;
    };
    return m_runtimeEnabled &&
        m_runtimeConfig.particleSpecies == m_draftConfig.particleSpecies &&
        m_runtimeConfig.totalGasCount == m_draftConfig.totalGasDensity &&
        sameFloat(m_runtimeConfig.ionizationFraction, m_draftConfig.ionizationFraction) &&
        sameFloat(m_runtimeConfig.electronTemperatureEv, m_draftConfig.electronTemperature) &&
        sameFloat(m_runtimeConfig.ionTemperatureEv, m_draftConfig.ionTemperature) &&
        sameFloat(m_runtimeConfig.neutralTemperatureK, m_draftConfig.neutralTemperature) &&
        m_runtimeConfig.selectedSpawnSelectionIndex == m_draftConfig.spawnSelectionIndex &&
        m_runtimeConfig.gridLayout == m_draftConfig.gridLayout;
}

void AtomicParticlesSimWorkspace::clearRuntime() {
    clearFieldDebug();
    m_paused = true;
    m_runtimeEnabled = false;
    m_activeMarkerCount = 0;
    m_elapsedSimulationTime = 0.0f;
    m_runtimeConfig = RuntimeConfig{};
    m_radii.clear();
    m_colors.clear();
    if (m_particleSystem) m_particleSystem->setActiveParticleCount(0);
}

bool AtomicParticlesSimWorkspace::applyRuntimeConfig() {
    if (!m_particleSystem) return false;

    RuntimeConfig resolved;

    if (!resolveRuntimeConfig(resolved))
        return false;

    SpawnDensityRegion3D spawnRegion;

    if (!m_spawnDensityGrid.selection(
        m_baseVoxelGrid,
        resolved.selectedSpawnSelectionIndex,
        spawnRegion))
        return false;

    // Validation above preserves the paused preview on invalid draft settings.
    // Once buffers are mutated, do not expose a partially configured runtime.
    m_runtimeEnabled = false;
    m_activeMarkerCount = 0;
    if (!m_particleSystem->setActiveParticleCount(resolved.activeMarkerCount))
        return false;

    if (!m_particleSystem->resetInBounds(
        ParticleSystem::CNFG_RANDOM_RESTART,
        make_float3(spawnRegion.minimum.x, spawnRegion.minimum.y, spawnRegion.minimum.z),
        make_float3(spawnRegion.maximum.x, spawnRegion.maximum.y, spawnRegion.maximum.z),
        resolved.placementRadius,
        kResetSeed)) return false;

    resolved.selectedSpawnVolumeM3 = spawnRegion.volumeM3;
    m_runtimeConfig = resolved;

    if (!configureRuntimeVisuals())
        return false;

    m_activeMarkerCount = resolved.activeMarkerCount;

    return true;
}

bool AtomicParticlesSimWorkspace::handleInput(
    const WorkspaceInputEvent& input,
    WorkspaceServices& services) {

    if (!m_active || !services.arbiter) return false;
    if (!m_arbiter) m_arbiter = services.arbiter;

    switch (services.arbiter->getApplicationLayer()) {
    case TheArbiter::ApplicationLayer::DOMAIN_SELECTION:
        return handleLayer1Input(input, services);

    case TheArbiter::ApplicationLayer::WORKSPACE_CONFIGURATION:
        return handleLayer2Input(input, services);

    case TheArbiter::ApplicationLayer::ACTIVE_WORKSPACE:
        return handleLayer3Input(input, services);

    case TheArbiter::ApplicationLayer::GLOBAL_SHELL:
    default:
        return false;
    }
}

bool AtomicParticlesSimWorkspace::handleLayer1Input(
    const WorkspaceInputEvent& input,
    WorkspaceServices& services) {

    switch (input.action) {
    case WorkspaceInputAction::Previous:
        moveLayer1Cursor(-1);
        return true;

    case WorkspaceInputAction::Next:
        moveLayer1Cursor(+1);
        return true;

    case WorkspaceInputAction::Decrease:
        adjustLayer1Value(-1, services);
        return true;

    case WorkspaceInputAction::Increase:
        adjustLayer1Value(+1, services);
        return true;

    case WorkspaceInputAction::Activate:
        if (m_layer1Selection == Layer1Row::WorkspaceSelection) {
            adjustLayer1Value(+1, services);
            return true;
        }

        if (m_layer1Selection == Layer1Row::Configure) {

            if (m_draftConfig.gridLayout == GridLayout::Dynamic) {
                m_statusLine = "DYNAMIC GRID is unavailable for this pass.";
                m_statusTone = WorkspaceStatusTone::Warning;
                return true;
            }

            if (m_draftConfig.simulationBoundary == SimulationBoundary::OpenReservoir) {
                m_statusLine = "OPEN_RESERVOIR boundary unavailable for this pass.";
                m_statusTone = WorkspaceStatusTone::Warning;
                return true;
            }

            clearRuntime();
            m_layer2Selection = Layer2Row::ParticleSpecies;
            m_textEntry.cancel();

            m_statusLine = "READY: ATOMIC_PARTICLES workspace configuration.";
            m_statusTone = WorkspaceStatusTone::Ready;

            services.arbiter->setApplicationLayer(
                TheArbiter::ApplicationLayer::WORKSPACE_CONFIGURATION
            );

            return true;
        }

        return true;

    case WorkspaceInputAction::Back:
        services.arbiter->requestReturnToGlobalShell(
            TheArbiter::WorkspaceDomain::MULPHY_SIM
        );
        return true;

    case WorkspaceInputAction::RawKey:
    case WorkspaceInputAction::None:
    default:
        return false;
    }
}

bool AtomicParticlesSimWorkspace::handleLayer2Input(
    const WorkspaceInputEvent& input,
    WorkspaceServices& services) {

    if (m_textEntry.isActive())
        return handleLayer2TextEntry(input);

    switch (input.action) {

    case WorkspaceInputAction::Previous:
        moveLayer2Cursor(-1);
        return true;

    case WorkspaceInputAction::Next:
        moveLayer2Cursor(+1);
        return true;

    case WorkspaceInputAction::Decrease:
        adjustLayer2Value(-1);
        return true;

    case WorkspaceInputAction::Increase:
        adjustLayer2Value(+1);
        return true;

    case WorkspaceInputAction::Activate:
        if (m_layer2Selection == Layer2Row::TotalGasDensity) {
            beginGasDensityEntry();
            return true;
        }

        if (m_layer2Selection == Layer2Row::RunSimulation) {

            m_paused = true;
            const bool resumeExisting = runtimeMatchesDraft();

            if (resumeExisting || applyRuntimeConfig()) {
                m_subLayers.reset();

                if (!resumeExisting) {
                    m_elapsedSimulationTime = 0.0f;
                    clearFieldDebug();
                }

                m_paused = false;
                m_runtimeEnabled = true;

                m_statusLine = "STATUS: RUNNING";
                m_statusTone = WorkspaceStatusTone::Ready;

                services.arbiter->setApplicationLayer(
                    TheArbiter::ApplicationLayer::ACTIVE_WORKSPACE
                );
            }
            else {
                m_statusLine = "PLASMA RUNTIME CONFIGURATION INVALID";
                m_statusTone = WorkspaceStatusTone::Warning;
            }

            return true;
        }

        return true;

    case WorkspaceInputAction::Back:
        m_textEntry.cancel();
        m_statusLine = "READY: MULTPHY_SIM WORKSPACE.";
        m_statusTone = WorkspaceStatusTone::Ready;
        services.arbiter->setApplicationLayer(
            TheArbiter::ApplicationLayer::DOMAIN_SELECTION
        );

        return true;

    case WorkspaceInputAction::RawKey:
    case WorkspaceInputAction::None:
    default:
        return false;
    }
}

bool AtomicParticlesSimWorkspace::handleLayer3Input(
    const WorkspaceInputEvent& input,
    WorkspaceServices& services) {

    if (input.action == WorkspaceInputAction::TogglePanel) {
        if (!input.repeated) {
            cancelInput(services);
            m_subLayers.togglePanel();
        }
        return true;
    }
    if (m_subLayers.handlePanelInput(input)) return true;

    if (handleFieldDebugKey(input, services)) return true;
    if (m_layer3CameraView == Layer3CameraView::Free &&
        setFreeMovementKey(input.action, true)) return true;

    switch (input.action) {

    case WorkspaceInputAction::TogglePause:
        if (input.repeated) return true;

        if (!m_runtimeEnabled) return true;
        m_paused = !m_paused;

        m_statusLine =
            m_paused
            ? "STATUS: PAUSED"
            : "STATUS: RUNNING";

        m_statusTone =
            m_paused
            ? WorkspaceStatusTone::Neutral
            : WorkspaceStatusTone::Ready;

        return true;

    case WorkspaceInputAction::Back:

        leaveRuntimeCamera(services);
        m_subLayers.hidePanel();
        m_testFireMode = false;
        m_paused = true;
        m_statusLine = "PAUSED: E on RUN SIM resumes; changed settings restart on E.";
        m_statusTone = WorkspaceStatusTone::Neutral;

        services.arbiter->setApplicationLayer(TheArbiter::ApplicationLayer::WORKSPACE_CONFIGURATION);

        return true;

    default:
        return false;
    }
}

void AtomicParticlesSimWorkspace::renderConfiguredGrid(
    WorkspaceServices& services, GridLayout layout) const {

    if (!services.renderer)
        return;

    EuclidRenderer::UniformGrid grid;
    const auto collisionGrid = m_particleSystem->getGridSize();

    grid.dimensions = ivec3(
        collisionGrid.x,
        collisionGrid.y, 
        collisionGrid.z
    );
    
    grid.origin = m_baseVoxelGrid.origin;
    grid.cellSize = vec3(m_simulationBoxSizeM) / vec3(grid.dimensions);
    grid.majorEvery = services.renderer->getGridMajorEvery();

    EuclidRenderer::GridDisplay display;

    display.boundary = true;
    display.majorGrid = layout == GridLayout::MajorGrid;
    display.minorGrid = false;
    display.axes = false;

    services.renderer->drawUniformGrid(grid, display);
}

void AtomicParticlesSimWorkspace::renderSelectedSpawnRegion(WorkspaceServices& services) const {
    if (!services.renderer) return;

    SpawnDensityRegion3D selectedRegion;
    if (!m_spawnDensityGrid.selection(
        m_baseVoxelGrid,
        m_draftConfig.spawnSelectionIndex,
        selectedRegion)) {

        return;
    }

    // Whole-domain preview has no 2x2x2 constituent geometry.
    if (m_draftConfig.spawnSelectionIndex == 0) {
        services.renderer->drawHighlightedVoxel(selectedRegion.center, selectedRegion.halfExtent, 4.0f);
        return;
    }

    // Draw each constituent base voxel so the 2x2x2 physical subdivision
    // remains visible, then reinforce the continuous composite boundary.
    for (const SpatialVoxelRegion& baseVoxel :
        selectedRegion.constituentBaseVoxels) {
        services.renderer->drawHighlightedVoxel(
            baseVoxel.center,
            baseVoxel.halfExtent,
            2.0f
        );
    }

    services.renderer->drawHighlightedVoxel(
        selectedRegion.center,
        selectedRegion.halfExtent,
        4.0f
    );
}

void AtomicParticlesSimWorkspace::beginGasDensityEntry() {
    if (m_textEntry.beginUnsignedInteger(
        "TOTAL GAS DESNITY",
        0,
        kParticleCapacity,
        m_draftConfig.totalGasDensity)) {

        m_statusLine = "ENTER TOTAL GAS DENSITY; E/Enter commits.";
        m_statusTone = WorkspaceStatusTone::Neutral;
    }
}

WorkspacePresentation
AtomicParticlesSimWorkspace::buildLayer1Presentation() const {

    WorkspacePresentation p;

    p.panelVisible = true;
    p.workspaceName = "LAYER 1 -> ATOMIC_PARTICLES WORKSPACE CONFIG";
    p.layerLabel = "MODE: ATOMIC_PARTICLES";

    WorkspacePanelSection section;

    section.rows.push_back(makeRow(
        "[1]: MULPHY_SIM SELECTION",
        "ATOMIC_PARTICLES",
        m_layer1Selection == Layer1Row::WorkspaceSelection
    ));

    section.rows.push_back(makeRow(
        "[2]: GRID LAYOUT",
        gridLayoutName(),
        m_layer1Selection == Layer1Row::GridLayout
    ));

    section.rows.push_back(makeRow(
        "[3]: SIMULATION BOX BOUNDARY",
        simulationBoundaryName(),
        m_layer1Selection == Layer1Row::SimulationBoundary
    ));

    section.rows.push_back(makeRow(
        "[4]: SIM SPACE MEDIUM",
        simSpaceMediumName(),
        m_layer1Selection == Layer1Row::SimSpaceMedium
    ));

    section.rows.push_back(makeRow(
        "[5]: PRESS E TO CONFIGURE WORKSPACE",
        "",
        m_layer1Selection == Layer1Row::Configure
    ));

    p.sections.push_back(section);

    p.statusLine = m_statusLine;
    p.statusTone = m_statusTone;
    p.footerLine1 = "W/S: Select row    A/D: Change value    E: Configure";
    p.footerLine2 = "Q: Return to Global Shell    ESC: Exit";
    return p;
}

WorkspacePresentation
AtomicParticlesSimWorkspace::buildLayer2Presentation() const {

    WorkspacePresentation p;

    p.panelVisible = true;
    p.workspaceName = "LAYER 2 -> MULTIPHYSICS_SIM CONFIGURATION";
    p.layerLabel = "MODE: MULTIPHYSICS_SIM";

    WorkspacePanelSection section;

    section.rows.push_back(makeRow(
        "[1]: PARTICLE SPECIES",
        particleSpeciesName(),
        m_layer2Selection == Layer2Row::ParticleSpecies
    ));

    // Density row supports interactive text entry
    if (m_textEntry.isActive() && m_layer2Selection ==
        Layer2Row::TotalGasDensity) {

        section.rows.push_back(makeRow(
            "[2]: TOTAL GAS DENSITY",
            ":=" + m_textEntry.getBuffer() + "m⁻³",
            true
        ));
    }
    else {
        section.rows.push_back(makeRow(
            "[2]: TOTAL GAS DENSITY",
            to_string(m_draftConfig.totalGasDensity) + "m^{-3}",
            m_layer2Selection == Layer2Row::TotalGasDensity
        ));
    }

    {
        ostringstream value;
        value
            << fixed
            << setprecision(2)
            << m_draftConfig.ionizationFraction;

        section.rows.push_back(makeRow(
            "[3]: IONIZATION",
            value.str(),
            m_layer2Selection == Layer2Row::IonizationFraction
        ));
    }

    section.rows.push_back(makeRow(
        "[4]: ELECTRON TEMP",
        to_string(m_draftConfig.electronTemperature) + "eV",
        m_layer2Selection == Layer2Row::ElectronTemperature
    ));

    section.rows.push_back(makeRow(
        "[5]: ION TEMP",
        to_string(m_draftConfig.ionTemperature) + "eV",
        m_layer2Selection == Layer2Row::IonTemperature
    ));

    section.rows.push_back(makeRow(
        "[6]: NEUTRAL TEMP",
        to_string(m_draftConfig.neutralTemperature) + " K",
        m_layer2Selection == Layer2Row::NeutralTemperature
    ));

    section.rows.push_back(makeRow(
        "[7]: SELECT VOXEL SPAWN",
        spawnSelectionText(m_draftConfig.spawnSelectionIndex),
        m_layer2Selection == Layer2Row::VoxelSpawn
    ));

    section.rows.push_back(makeRow(
        "[8]: PRESS E TO RUN SIM",
        "",
        m_layer2Selection == Layer2Row::RunSimulation
    ));

    p.sections.push_back(section);

    if (m_statusTone != WorkspaceStatusTone::Ready) {
        p.statusLine = m_statusLine;
        p.statusTone = m_statusTone;
    }

    // Plasma population summary
    p.postStatusLines.push_back(
        "NEUTRAL " + string(particleSpeciesName()) +
        ": " + to_string(neutralCount())
    );

    p.postStatusLines.push_back(
        "IONIZED " + string(particleSpeciesName()) +
        ": " + to_string(ionCount())
    );

    p.postStatusLines.push_back("FREE ELECTRONS: " + to_string(electronCount()));
    p.postStatusLines.push_back("--------------------");
    p.postStatusLines.push_back(
        string(particleSpeciesName()) + " TOTAL: " +
        to_string(m_draftConfig.totalGasDensity)
    );

    p.postStatusLines.push_back(
        "SIM MARKERS: " + to_string(requestedMarkerCount()) +
        "/" + to_string(kParticleCapacity)
    );

    p.footerLine1 = "W/S: Select row    A/D: Change value    E: Activate / Enter";
    p.footerLine2 = "Q: Return to Layer 1    ESC: Exit";

    return p;
}

WorkspacePresentation
AtomicParticlesSimWorkspace::buildLayer3Presentation() const {
    auto p = m_subLayers.buildPresentation("ATOMIC_PARTICLES MODE");
    p.runtimeStatus = buildRuntimeStatus();
    return p;
}

WorkspaceRuntimeStatus AtomicParticlesSimWorkspace::buildRuntimeStatus() const {

    WorkspaceRuntimeStatus status;
    status.visible = true;

    status.titleLine =
        "LAYER 3 -> SIMULATION RUNTIME "
        "(ATOMIC_PARTICLES)";

    status.contextLine = m_subLayers.context("ATOMIC_PARTICLES");

    const bool running = m_runtimeEnabled && !m_paused;

    status.objectLine = "SIM MARKERS: " +
        to_string(m_runtimeConfig.activeMarkerCount) + "/" +
        to_string(kParticleCapacity) + "        STATUS: " +
        (running
            ? "RUNNING"
            : "PAUSED");

    status.objectTone =
        running
        ? WorkspaceStatusTone::Ready
        : WorkspaceStatusTone::Neutral;

    status.helpLine = m_subLayers.help() + "    SPACE: PAUSE    Q: BACK    RIGHT CLICK: CAMERA / TEST MENU";
    appendFieldDebugStatus(status);

    return status;
}

WorkspacePresentation AtomicParticlesSimWorkspace::buildPresentation() const {

    if (!m_arbiter) return {};

    switch (m_arbiter->getApplicationLayer()) {

    case TheArbiter::ApplicationLayer::DOMAIN_SELECTION:
        return buildLayer1Presentation();

    case TheArbiter::ApplicationLayer::WORKSPACE_CONFIGURATION:
        return buildLayer2Presentation();

    case TheArbiter::ApplicationLayer::ACTIVE_WORKSPACE:
        return buildLayer3Presentation();

    case TheArbiter::ApplicationLayer::GLOBAL_SHELL:
    default:
        return {};
    }
}

void AtomicParticlesSimWorkspace::moveLayer1Cursor(int direction) {
    if (direction == 0) return;
    const int count = static_cast<int>(Layer1Row::Count);
    const int current = static_cast<int>(m_layer1Selection);
    const int step = direction < 0 ? -1 : +1;
    m_layer1Selection = static_cast<Layer1Row>(
        (current + step + count) % count
        );
}

void AtomicParticlesSimWorkspace::moveLayer2Cursor(int direction) {
    if (direction == 0) return;
    const int count = static_cast<int>(Layer2Row::Count);
    const int current = static_cast<int>(m_layer2Selection);
    const int step = direction < 0 ? -1 : +1;
    m_layer2Selection = static_cast<Layer2Row>((current + step + count) % count);
}

void AtomicParticlesSimWorkspace::adjustLayer1Value(int direction, WorkspaceServices& services) {

    if (direction == 0) return;

    const int step = direction < 0 ? -1 : +1;

    switch (m_layer1Selection) {
        // MULTIPHYSICS_SIM <-> PARTICLE_SIM
        //
        // This preserves the same cartridge-selection
        // behavior PARTICLE_SIM already uses.
    case Layer1Row::WorkspaceSelection:
        if (services.arbiter) {

            services.arbiter->cycleMulphyWorkspace(direction);
        }
        return;

    case Layer1Row::GridLayout: {
        const int count = static_cast<int>(GridLayout::Count);
        const int current = static_cast<int>(m_draftConfig.gridLayout);
        m_draftConfig.gridLayout = static_cast<GridLayout>((current + step + count) % count);
        break;
    }
    case Layer1Row::SimulationBoundary: {
        const int count =
            static_cast<int>(SimulationBoundary::Count);

        const int current =
            static_cast<int>(m_draftConfig.simulationBoundary);

        m_draftConfig.simulationBoundary =
            static_cast<SimulationBoundary>((current + step + count) % count);

        break;
    }
    case Layer1Row::SimSpaceMedium: {
        const int count = static_cast<int>(SimSpaceMedium::Count);
        const int current = static_cast<int>(m_draftConfig.simSpaceMedium);

        m_draftConfig.simSpaceMedium = 
            static_cast<SimSpaceMedium>((current + step + count) % count);

        break;
    }
    case Layer1Row::Configure:
    case Layer1Row::Count:
    default:
        return;
    }
}

void AtomicParticlesSimWorkspace::adjustLayer2Value(int direction) {
    if (direction == 0) return;

    const int step = direction < 0 ? -1 : +1;

    switch (m_layer2Selection) {

    case Layer2Row::ParticleSpecies: {
        const int count = static_cast<int>(ParticleSpecies::Count);
        const int current = static_cast<int>(m_draftConfig.particleSpecies);
        m_draftConfig.particleSpecies = static_cast<ParticleSpecies>(
            (current + step + count) % count
            );
        break;
    }
    case Layer2Row::IonizationFraction:
        m_draftConfig.ionizationFraction = std::clamp(
            m_draftConfig.ionizationFraction + static_cast<float>(step) * 0.01f,
            0.0f,
            1.0f
        );
        break;

    case Layer2Row::ElectronTemperature:
        m_draftConfig.electronTemperature = std::max(
            0.0f,
            m_draftConfig.electronTemperature + step * 0.1f
        );
        break;

    case Layer2Row::IonTemperature:
        m_draftConfig.ionTemperature = std::max(
            0.0f,
            m_draftConfig.ionTemperature + step * 0.1f
        );
        break;

    case Layer2Row::NeutralTemperature:
        m_draftConfig.neutralTemperature = std::max(
            0.0f,
            m_draftConfig.neutralTemperature + step * 0.1f
        );
        break;

    case Layer2Row::VoxelSpawn: {

        const unsigned int count =
            m_spawnDensityGrid.selectionCount(m_baseVoxelGrid);

        if (count == 0) break;
        const int current = static_cast<int>(m_draftConfig.spawnSelectionIndex);

        m_draftConfig.spawnSelectionIndex = static_cast<unsigned int>(
            (current + step + static_cast<int>(count)) % static_cast<int>(count)
            );
        break;
    }

    case Layer2Row::RunSimulation:
    case Layer2Row::Count:
    default:
        return;
    }
}

void AtomicParticlesSimWorkspace::refreshLayer1Status() {

    // DYNAMIC grid is still unavailable.
    if (m_draftConfig.gridLayout == GridLayout::Dynamic) {

        m_statusLine = "DYNAMIC GRID is unavailable for this pass.";
        m_statusTone = WorkspaceStatusTone::Warning;

        return;
    }

    // OPEN_RESERVOIR is reserved for a future pass.
    if (m_draftConfig.simulationBoundary == SimulationBoundary::OpenReservoir) {

        m_statusLine = "OPEN_RESERVOIR boundary unavailable for this pass.";

        m_statusTone =WorkspaceStatusTone::Warning;

        return;
    }

    m_statusLine = "READY: ATOMIC_PARTICLES WORKSPACE.";

    m_statusTone = WorkspaceStatusTone::Ready;
}

bool AtomicParticlesSimWorkspace::handleLayer2TextEntry(const WorkspaceInputEvent& input) {

    if (input.action == WorkspaceInputAction::Back ||
        input.rawKey == 'q' || input.rawKey == 'Q') {

        m_textEntry.cancel();
        m_statusLine = "TOTAL GAS DENSITY entry cancelled.";
        m_statusTone = WorkspaceStatusTone::Neutral;

        return true;
    }

    const TextEntryAction action = m_textEntry.handleRawKey(input.rawKey);

    switch (action) {

    case TextEntryAction::Committed: {
        unsigned int value = 0;

        if (m_textEntry.tryGetCommittedUnsigned(value)) {
            m_draftConfig.totalGasDensity = value;
            m_statusLine = "READY: TOTAL GAS DENSITY committed.";
            m_statusTone = WorkspaceStatusTone::Ready;
        }
        else {
            m_statusLine = "TOTAL GAS DENSITY was not committed.";
            m_statusTone = WorkspaceStatusTone::Warning;
        }

        break;
    }
    case TextEntryAction::Cancelled:
        m_statusLine = "TOTAL GAS DENSITY entry cancelled.";
        m_statusTone = WorkspaceStatusTone::Neutral;
        break;


    case TextEntryAction::Rejected:
        m_statusLine = m_textEntry.getStatusMessage();
        m_statusTone = WorkspaceStatusTone::Warning;
        break;

    case TextEntryAction::Changed:
        m_statusLine = "ENTER TOTAL GAS DENSITY; E/ENTER commits.";
        m_statusTone = WorkspaceStatusTone::Neutral;
        break;

    case TextEntryAction::None:
    default:
        break;
    }

    return true;
}

unsigned int AtomicParticlesSimWorkspace::ionCount() const {
    const float value = static_cast<float>(m_draftConfig.totalGasDensity) *
        m_draftConfig.ionizationFraction;

    return static_cast<unsigned int>(round(value));
}

unsigned int AtomicParticlesSimWorkspace::neutralCount() const {
    return m_draftConfig.totalGasDensity - ionCount();
}

unsigned int AtomicParticlesSimWorkspace::electronCount() const {
    return ionCount();
}

unsigned int AtomicParticlesSimWorkspace::requestedMarkerCount() const {
    return neutralCount() + ionCount() + electronCount();
}

string AtomicParticlesSimWorkspace::spawnSelectionText(
    unsigned int selectionIndex) const {
    // Presentation label only: existing world-space units remain unchanged.
    if (selectionIndex == 0)
        return "[" + std::to_string(static_cast<int>(m_simulationBoxSizeM)) + " MICRO METER]^3";
    if (selectionIndex == 1) {
        return "VOXEL_CENTER";
    }

    const unsigned int voxelId =
        selectionIndex - 2;

    ostringstream stream;

    stream
        << "VOXEL_"
        << setw(3)
        << setfill('0')
        << voxelId;

    return stream.str();
}

float AtomicParticlesSimWorkspace::selectedSpeciesRenderRadius() const {

    switch (m_draftConfig.particleSpecies) {

    case ParticleSpecies::Hydrogen:
        return kHydrogenRadius;

    case ParticleSpecies::Helium:
        return kHeliumRadius;

    case ParticleSpecies::Argon:
    default:
        return kArgonRadius;
    }
}

const char* AtomicParticlesSimWorkspace::gridLayoutName() const {
    switch (m_draftConfig.gridLayout) {
    case GridLayout::MajorGrid: return "MAJOR_GRID";
    case GridLayout::Dynamic: return "DYNAMIC";
    case GridLayout::None:
    default:
        return "NONE";
    }
}

const char* AtomicParticlesSimWorkspace::simulationBoundaryName() const {
    switch (m_draftConfig.simulationBoundary) {

    case SimulationBoundary::OpenReservoir:
        return "OPEN_RESERVOIR";

    case SimulationBoundary::Closed:
    default:
        return "CLOSED";
    }
}

const char* AtomicParticlesSimWorkspace::simSpaceMediumName() const {
    switch (m_draftConfig.simSpaceMedium) {

    case SimSpaceMedium::Air:
        return "AIR";

    case SimSpaceMedium::Vacuum:
    default:
        return "VACUUM";
    }
}

const char* AtomicParticlesSimWorkspace::particleSpeciesName() const {
    switch (m_draftConfig.particleSpecies) {

    case ParticleSpecies::Hydrogen:
        return "HYDROGEN";

    case ParticleSpecies::Helium:
        return "HELIUM";

    case ParticleSpecies::Argon:
    default:
        return "ARGON";
    }
}
