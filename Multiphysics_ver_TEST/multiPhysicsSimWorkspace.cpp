#include "multiPhysicsSimWorkspace.h"
#include "SimulationPreset.h"
#include "rendererEM_Euclid.h"

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


bool MultiPhysicsSimWorkspace::initialize(
    WorkspaceServices& services) {

    if (m_initialized)
        return true;

    if (!services.renderer ||
        !services.arbiter) {
        return false;
    }

    m_arbiter = services.arbiter;

    syncSimulationDomain(services);

    m_initialized = true;

    return true;
}


void MultiPhysicsSimWorkspace::syncSimulationDomain(
    WorkspaceServices& services) {

    if (!services.renderer)
        return;

    const int boxSize =
        services.renderer->getSimBoxSize();

    const SimulationPreset* preset =
        findSimulationPreset(boxSize);

    if (!preset ||
        !preset->enabled) {
        return;
    }

    // The Layer-0 committed preset is authoritative.
    m_simulationBoxSize =
        static_cast<float>(boxSize);

    m_collisionGridDimension =
        preset->collisionGridDim;

    // Preserve the shared physical 8x8x8 workspace geometry.
    m_baseVoxelGrid.dimensions =
        ivec3(8);

    m_baseVoxelGrid.origin =
        vec3(
            -m_simulationBoxSize * 0.5f
        );

    m_baseVoxelGrid.voxelEdgeM =
        m_simulationBoxSize / 8.0f;
}


SimulationDomainState
MultiPhysicsSimWorkspace::simulationDomainState() const {

    SimulationDomainState state;

    state.physicalGrid =
        m_baseVoxelGrid;

    state.collisionDimensions =
        ivec3(m_collisionGridDimension);

    state.collisionOrigin =
        vec3(
            -m_simulationBoxSize * 0.5f
        );

    state.collisionCellSize =
        vec3(
            m_simulationBoxSize /
            static_cast<float>(
                m_collisionGridDimension
                )
        );

    const unsigned int dimension =
        static_cast<unsigned int>(
            m_collisionGridDimension
            );

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

    if (!m_active ||
        !frame.displayEnabled ||
        !services.renderer ||
        !services.arbiter) {
        return;
    }

    // This workspace currently exists only at Layer 1.
    if (services.arbiter->getApplicationLayer() !=
        TheArbiter::ApplicationLayer::DOMAIN_SELECTION) {
        return;
    }

    renderUniformGrid(services);
}


void MultiPhysicsSimWorkspace::renderUniformGrid(
    WorkspaceServices& services) const {

    if (!services.renderer)
        return;

    EuclidRenderer::UniformGrid grid;

    grid.dimensions =
        ivec3(m_collisionGridDimension);

    grid.origin =
        vec3(
            -m_simulationBoxSize * 0.5f
        );

    grid.cellSize =
        vec3(
            m_simulationBoxSize /
            static_cast<float>(
                m_collisionGridDimension
                )
        );

    grid.majorEvery =
        kMajorGridEvery;

    EuclidRenderer::GridDisplay display;

    display.boundary = true;
    display.majorGrid = true;
    display.minorGrid = false;
    display.axes = false;

    services.renderer->drawUniformGrid(
        grid,
        display
    );
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

        services.arbiter->
            cycleMulphyWorkspace(-1);

        return true;


    case WorkspaceInputAction::Increase:

        services.arbiter->
            cycleMulphyWorkspace(+1);

        return true;


    case WorkspaceInputAction::Activate:

        // E on the selector behaves like D:
        // MULTIPHYSICS_SIM -> PARTICLE_SIM.
        services.arbiter->
            cycleMulphyWorkspace(+1);

        return true;


    case WorkspaceInputAction::Previous:
    case WorkspaceInputAction::Next:

        // Only one Layer-1 row exists.
        return true;


    case WorkspaceInputAction::Back:

        services.arbiter->
            requestReturnToGlobalShell(
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
        "READY: MULTIPHYSICS_SIM INTEGRATION CANVAS.";

    p.statusTone =
        WorkspaceStatusTone::Ready;

    p.footerLine1 =
        "A/D: Change workspace    E: Next workspace";

    p.footerLine2 =
        "Q: Return to Global Shell    ESC: Exit";

    return p;
}