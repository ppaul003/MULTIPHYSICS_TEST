#ifndef NDMSM_MULTIPHYSICS_SIM_WORKSPACE_H
#define NDMSM_MULTIPHYSICS_SIM_WORKSPACE_H

#include "IWorkspaceEM.h"
#include "TheArbiterEM.h"

class MultiPhysicsSimWorkspace final : public IWorkspace {
public:

    bool initialize(
        WorkspaceServices& services
    ) override;

    void syncSimulationDomain(
        WorkspaceServices& services
    );

    SimulationDomainState
        simulationDomainState() const;

    void enter(
        WorkspaceServices& services
    ) override;

    void exit(
        WorkspaceServices& services
    ) override;

    void update(
        const WorkspaceFrameContext& frame,
        WorkspaceServices& services
    ) override;

    void render(
        const WorkspaceFrameContext& frame,
        WorkspaceServices& services
    ) override;

    bool handleInput(
        const WorkspaceInputEvent& input,
        WorkspaceServices& services
    ) override;

    WorkspacePresentation
        buildPresentation() const override;

private:

    void renderUniformGrid(
        WorkspaceServices& services
    ) const;

private:

    TheArbiter* m_arbiter = nullptr;

    // Geometry inherited from Layer 0.
    float m_simulationBoxSize = 4.0f;
    int m_collisionGridDimension = 64;

    SpatialVoxelGrid3D m_baseVoxelGrid;

    static constexpr int kMajorGridEvery = 8;

    bool m_initialized = false;
    bool m_active = false;
};

#endif