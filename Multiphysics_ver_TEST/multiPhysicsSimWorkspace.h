#ifndef NDMSM_MULTIPHYSICS_SIM_WORKSPACE_H
#define NDMSM_MULTIPHYSICS_SIM_WORKSPACE_H

#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <vector_functions.h>
#include <vector_types.h>

#include "IWorkspaceEM.h"
#include "TextEntry.h"
#include "TheArbiterEM.h"
#include "particleSystem.h"

class MultiPhysicsSimWorkspace final : public IWorkspace {
public:

    bool initialize(WorkspaceServices& services) override;
    void syncSimulationDomain(WorkspaceServices& services);
    SimulationDomainState simulationDomainState() const;
    void enter(WorkspaceServices& services) override;
    void exit(WorkspaceServices& services) override;
    void update(const WorkspaceFrameContext& frame, WorkspaceServices& services) override;
    void render(const WorkspaceFrameContext& frame, WorkspaceServices& services) override;
    bool handleInput(const WorkspaceInputEvent& input, WorkspaceServices& services) override;

    WorkspacePresentation buildPresentation() const override;

private:
    void renderConfiguredGrid(WorkspaceServices& services) const;

private:
    static constexpr unsigned int kParticleCapacity = 16384;
    static constexpr unsigned int kGridSize = 64;
    static constexpr unsigned int kMajorGridEvery = 8;
    
    unsigned int m_capacity = kParticleCapacity;
    unsigned int m_activeCount = 0;
    uint3 m_gridDimensions = make_uint3(kGridSize, kGridSize, kGridSize);

    std::unique_ptr<ParticleSystem> m_particleSystem;

    TheArbiter* m_arbiter = nullptr;
    SpatialVoxelGrid3D m_baseVoxelGrid;

    // Geometry inherited from Layer 0.
    
    int m_collisionGridDimension = 64;

    bool m_initialized = false;
    bool m_active = false;

    float m_simulationBoxSize = 32.0f;
    float m_elapsedSimulationTime = 0.0f;
};

#endif