#include "multiPhysicsSimWorkspace.h"
#include "SimulationPreset.h"
#include "rendererEM_Euclid.h"
#include "CameraEM.h"

#include <paramgl.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

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


bool MultiPhysicsSimWorkspace::initialize(WorkspaceServices& services) {
    if (m_initialized) return true;

    if (!services.renderer ||
        !services.arbiter)
        return false;

    m_arbiter = services.arbiter;

    m_particleSystem = make_unique<ParticleSystem>(
        kParticleCapacity,
        m_gridDimensions,
        true
    );

    syncSimulationDomain(services);

    if (!m_particleSystem->setActiveParticleCount(0))
        return false;

    m_initialized = true;

    return true;
}


void MultiPhysicsSimWorkspace::syncSimulationDomain(
    WorkspaceServices& services) {

    if (!services.renderer ||
        !m_particleSystem)
        return;

    const float boxSize = static_cast<float>(services.renderer->getSimBoxSize());
    const auto* preset = findSimulationPreset(services.renderer->getSimBoxSize());

    if (!preset ||
        !preset->enabled ||
        services.renderer->getGridDimSize()
        != preset->collisionGridDim)
        return;

    const auto currentGrid = m_particleSystem->getGridSize();
    const uint dimension = static_cast<uint>(preset->collisionGridDim);

    if (boxSize == m_simulationBoxSize && currentGrid.x == dimension &&
        currentGrid.y == dimension && currentGrid.z == dimension)
        return;

    m_particleSystem->setSimulationDomain(
        boxSize,
        make_uint3(dimension, dimension, dimension)
    );

    m_simulationBoxSize = boxSize;
    m_baseVoxelGrid.dimensions = ivec3(8);
    m_baseVoxelGrid.origin = vec3(-boxSize * 0.5f);
    m_baseVoxelGrid.voxelEdgeM = boxSize / 8.0f;
}


SimulationDomainState
MultiPhysicsSimWorkspace::simulationDomainState() const {

    SimulationDomainState state;

    state.physicalGrid =
        m_baseVoxelGrid;

    state.collisionDimensions =
        ivec3(m_collisionGridDimension);

    state.collisionOrigin =
        vec3(-m_simulationBoxSize * 0.5f);

    state.collisionCellSize =
        vec3(m_simulationBoxSize / static_cast<float>(m_collisionGridDimension));

    const unsigned int dimension =
        static_cast<unsigned int>(m_collisionGridDimension);

    state.collisionCellCount =
        dimension *
        dimension *
        dimension;

    // This workspace intentionally has no particle or field runtime yet.
    state.collisionRadius = 0.0f;
    state.fieldCellCount = 0;
    state.fieldGeometryValid = false;

    return state;
}


void MultiPhysicsSimWorkspace::enter(
    WorkspaceServices& services) {

    syncSimulationDomain(services);

    if (!m_arbiter)
        m_arbiter = services.arbiter;

    m_active = true;
}


void MultiPhysicsSimWorkspace::exit(
    WorkspaceServices& services) {

    (void)services;

    m_active = false;
}


void MultiPhysicsSimWorkspace::update(
    const WorkspaceFrameContext& frame,
    WorkspaceServices& services) {

    (void)frame;
    (void)services;

    // Intentionally empty.
    //
    // MULTIPHYSICS_SIM is currently only an
    // integration canvas at Layer 1.
}


void MultiPhysicsSimWorkspace::render(
    const WorkspaceFrameContext& frame,
    WorkspaceServices& services) {

    if (!frame.displayEnabled ||
        !services.renderer ||
        !services.arbiter) {
        return;
    }

    switch (services.arbiter->getApplicationLayer()) {
    case TheArbiter::ApplicationLayer::DOMAIN_SELECTION:
        renderConfiguredGrid(services);
        return;

    case TheArbiter::ApplicationLayer::WORKSPACE_CONFIGURATION:
    case TheArbiter::ApplicationLayer::ACTIVE_WORKSPACE:
    case TheArbiter::ApplicationLayer::GLOBAL_SHELL:
    default:
        return;
    }
    // This workspace currently exists only at Layer 1.
    if (services.arbiter->getApplicationLayer() !=
        TheArbiter::ApplicationLayer::DOMAIN_SELECTION) {
        return;
    }
}

void MultiPhysicsSimWorkspace::renderConfiguredGrid(WorkspaceServices& services) const {

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
    grid.cellSize = vec3(m_simulationBoxSize) / vec3(grid.dimensions);
    grid.majorEvery = services.renderer->getGridMajorEvery();

    EuclidRenderer::GridDisplay display;

    display.boundary = true;
    display.majorGrid = true;
    display.minorGrid = false;
    display.axes = false;

    services.renderer->drawUniformGrid(grid, display);
}


bool MultiPhysicsSimWorkspace::handleInput(
    const WorkspaceInputEvent& input,
    WorkspaceServices& services) {

    if (!m_active ||
        !services.arbiter) {
        return false;
    }

    if (!m_arbiter)
        m_arbiter = services.arbiter;

    // MULTIPHYSICS_SIM currently has Layer 1 only.
    if (services.arbiter->getApplicationLayer() !=
        TheArbiter::ApplicationLayer::DOMAIN_SELECTION) {
        return false;
    }

    switch (input.action) {

    case WorkspaceInputAction::Decrease:

        services.arbiter->cycleMulphyWorkspace(-1);

        return true;


    case WorkspaceInputAction::Increase:

        services.arbiter->cycleMulphyWorkspace(+1);

        return true;


    case WorkspaceInputAction::Activate:

        // E on the selector behaves like D:
        // MULTIPHYSICS_SIM -> PARTICLE_SIM.
        services.arbiter->cycleMulphyWorkspace(+1);

        return true;


    case WorkspaceInputAction::Previous:
    case WorkspaceInputAction::Next:

        // Only one Layer-1 row exists.
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


WorkspacePresentation
MultiPhysicsSimWorkspace::buildPresentation() const {

    if (!m_arbiter)
        return {};

    if (m_arbiter->getApplicationLayer() !=
        TheArbiter::ApplicationLayer::DOMAIN_SELECTION) {
        return {};
    }

    WorkspacePresentation p;

    p.panelVisible = true;

    p.workspaceName =
        "LAYER 1 -> MULTIPHYSICS_SIM WORKSPACE";

    p.layerLabel =
        "MODE: MULTIPHYSICS_SIM";

    WorkspacePanelSection section;

    section.rows.push_back(
        makeRow(
            "[1]: MULPHY_SIM SELECTION",
            "MULTIPHYSICS_SIM",
            true
        )
    );

    p.sections.push_back(section);

    p.statusLine =
        "UNAVAILABLE: Workspace mode still under planning...";

    p.statusTone = WorkspaceStatusTone::Warning;

    p.footerLine1 =
        "A/D: Change workspace    E: Next workspace";

    p.footerLine2 =
        "Q: Return to Global Shell    ESC: Exit";

    return p;
}