#ifndef NDMSM_ATOMIC_PARTICLES_WORKSPACE_H
#define NDMSM_ATOMIC_PARTICLES_WORKSPACE_H

#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <vector_functions.h>
#include <vector_types.h>

#include "IWorkspaceEM.h"
#include "AtomicInitialConditions.h"
#include "RuntimeSubLayerTraversal.h"
#include "TextEntry.h"
#include "TheArbiterEM.h"
#include "particleSystem.h"
#include "DebugElectrodynamics.h"
#include "FieldDebugRenderer.h"
#include "fieldSystem.h"

class AtomicParticlesSimWorkspace final : public IWorkspace {
public:

    bool initialize(WorkspaceServices& services) override;
    void syncSimulationDomain(WorkspaceServices& services);
    // Called by the workspace host before OpenGL/CUDA context teardown.
    void releaseFieldResources() { m_fieldSystem.reset(); m_fieldMirrorReady = false; }
    SimulationDomainState simulationDomainState() const;
    void enter(WorkspaceServices& services) override;
    void exit(WorkspaceServices& services) override;

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

    WorkspacePresentation buildPresentation() const override;
    bool textEntryActive() const override;
    const AtomicInitialization::Population& runtimePopulation() const { return m_runtimeConfig; }
    WorkspaceMenuPresentation buildMenu() const override;
    bool handleMenuCommand(int command, WorkspaceServices& services) override;
    bool handlePointerInput(const WorkspacePointerEvent& input, WorkspaceServices& services) override;
    bool handleInputRelease(const WorkspaceInputEvent& input, WorkspaceServices& services) override;
    void cancelInput(WorkspaceServices& services) override;
    void renderOverlay(const WorkspaceFrameContext& frame, WorkspaceServices& services) override;

private:
    enum class Layer1Row {
        WorkspaceSelection = 0,
        InitialMaterialPhase,
        SimulationBoundary,
        SimSpaceMedium,
        Configure,
        Count
    };

    enum class Layer2Row {
        ParticleSpecies = 0,
        DensityType,
        Density,
        IonizationFraction,
        ElectronTemperature,
        GasInput,
        RunSimulation,
        Count
    };

    using DensityType = AtomicInitialization::DensityType;

    enum class InitialMaterialPhase { Plasma = 0, Gas, Liquid, Solid, Count };

    enum class GridLayout {
        None = 0,
        MajorGrid,
        Dynamic,
        Count
    };

    enum class SimulationBoundary {
        Closed = 0,
        OpenReservoir,
        Count
    };

    enum class SimSpaceMedium {
        Vacuum = 0,
        Air,
        Count
    };

    enum class ParticleSpecies {
        Hydrogen = 0,
        Helium,
        Argon,
        Count
    };

    enum class Layer3CameraView {
        Orbit = 0,
        Free
    };

    struct ParticleVisual {
        float renderRadius;
        float4 color;
        float emissiveStrength;
    };

    struct ParticlePhysics {
        double mass;
        double charge;
    };

    struct DraftConfig {

        // --- Layer 1 ---
        InitialMaterialPhase initialMaterialPhase = InitialMaterialPhase::Gas;
        GridLayout gridLayout = GridLayout::None;
        SimulationBoundary simulationBoundary = SimulationBoundary::Closed;
        SimSpaceMedium simSpaceMedium = SimSpaceMedium::Vacuum;

        // --- Layer 2 ---
        ParticleSpecies particleSpecies = ParticleSpecies::Argon;
        DensityType densityType = DensityType::Electron;
        double densityMantissa = 0.0;
        int densityExponent = 0;
        double ionizationFraction = 0.10;
        double electronTemperature = 0.0;
        double gasTemperature = 0.0;
    };

    struct RuntimeConfig : AtomicInitialization::Population {
        // Exact input snapshot for change-back/resume, including Layer 1.
        DraftConfig initialConditions;
        float speciesRadius = 0.0f;
        float placementRadius = 0.0f;
    };

    WorkspacePresentation buildLayer1Presentation() const;
    WorkspacePresentation buildLayer2Presentation() const;
    WorkspacePresentation buildLayer3Presentation() const;

    WorkspaceRuntimeStatus buildRuntimeStatus() const;

    bool handleLayer1Input(const WorkspaceInputEvent& input, WorkspaceServices& services);
    bool handleLayer2Input(const WorkspaceInputEvent& input, WorkspaceServices& services);
    bool handleLayer3Input(const WorkspaceInputEvent& input, WorkspaceServices& services);

    bool handleLayer2TextEntry(const WorkspaceInputEvent& input);

    bool runtimeMatchesDraft() const;
    void clearRuntime();
    bool applyRuntimeConfig();
    bool resolveRuntimeConfig(RuntimeConfig& resolved, std::string& error) const;
    bool configureRuntimeVisuals();

    float selectedSpeciesRenderRadius() const;

    void renderConfiguredGrid(WorkspaceServices& services, GridLayout layout) const;
    void beginDensityEntry();
    std::string densityNotation() const;

    void moveLayer1Cursor(int direction);
    void moveLayer2Cursor(int direction);

    void adjustLayer1Value(int direction, WorkspaceServices& services);
    void adjustLayer2Value(int direction);

    void refreshLayer1Status();

    void renderActivePlasmaMarkers(WorkspaceServices& services);

    enum class VectorView { Off, Electric, Magnetic, Current, CurlB, Count };
    enum class ScalarView { Off, ElectronDensity, ElectronTemperature, ChargeDensity, Count };
    enum DebugMenuCommand { MenuCameraView = 100, MenuFireMode, MenuClearDebug, MenuFieldBackend };

    bool layer3Active() const;
    void initializeFields();
    void clearFieldDebug();
    void refreshDiagnosticFields();
    bool mirrorDiagnosticFields();
    void updateFieldDebug(const WorkspaceFrameContext& frame, WorkspaceServices& services);
    void renderFieldDebug(WorkspaceServices& services);
    bool handleFieldDebugKey(const WorkspaceInputEvent& input, WorkspaceServices& services);
    bool setFreeMovementKey(WorkspaceInputAction action, bool pressed);
    void leaveRuntimeCamera(WorkspaceServices& services);
    void appendFieldDebugStatus(WorkspaceRuntimeStatus& status) const;

    const VectorField3D* selectedVectorField() const;
    const ScalarField3D* selectedScalarField() const;
    
    const char* vectorViewName() const;
    const char* scalarViewName() const;
    const char* initialMaterialPhaseName() const;
    const char* simulationBoundaryName() const;
    const char* simSpaceMediumName() const;
    const char* particleSpeciesName() const;

private:
    static constexpr float kElectronRadius = 0.0015f;
    static constexpr float kHydrogenRadius = 0.0040f;
    static constexpr float kHeliumRadius = 0.0047f;
    static constexpr float kArgonRadius = 0.0063f;

    float m_simulationBoxSizeWorld = 0.0f;

    static constexpr float kMaximumSupportedRadius = 0.0156f;
    static constexpr unsigned int kParticleCapacity = AtomicInitialization::kParticleCapacity;
    static constexpr unsigned int kMajorGridEvery = 8;
    static constexpr unsigned int kGridSize = 64;
    static constexpr unsigned int kResetSeed = 1973;

    std::unique_ptr<ParticleSystem> m_particleSystem;
    std::unique_ptr<FieldSystem> m_fieldSystem;
    FieldBackend m_fieldBackend = FieldBackend::CUDA;
    bool m_fieldMirrorReady = false;
    std::vector<float4> m_fieldGlyphPalette;
    std::string m_statusLine = "READY: MULTIPHY_SIM MODE ONLINE.";

    std::vector<float> m_radii;
    std::vector<float4> m_colors;

    TheArbiter* m_arbiter = nullptr;
    DraftConfig m_draftConfig;
    SpatialVoxelGrid3D m_baseVoxelGrid;
    SpawnDensityRegionGrid3D m_spawnDensityGrid;
    RuntimeConfig m_runtimeConfig;

    // Field sampling is independent of the fixed 8x8x8 spawn-density grid.
    // This stable member owns geometry borrowed by all diagnostic fields.
    SpatialVoxelGrid3D m_fieldVoxelGrid;
    ScalarField3D m_electronDensity, m_electronTemperature, m_chargeDensity;
    VectorField3D m_electricField, m_magneticField, m_currentDensity, m_curlMagneticField;
    std::vector<DiagnosticFieldSample> m_fieldVisualization;
    std::vector<glm::vec4> m_electricGlyphColors, m_magneticGlyphColors;
    DebugElectrodynamics m_debugElectrodynamics;
    VectorView m_vectorView = VectorView::Off;
    ScalarView m_scalarView = ScalarView::Off;
    VectorFieldRenderSettings m_vectorRenderSettings{VectorGlyphScale::LogMagnitude};
    DebugProjectileSpecies m_debugSpecies = DebugProjectileSpecies::Electron;
    Layer3CameraView m_layer3CameraView = Layer3CameraView::Orbit;

    bool m_testFireMode = false;
    bool m_fireClickCaptured = false;
    bool m_freeLookDragging = false;
    bool m_freeMovementKeys[4]{};
    std::string m_debugNotice;
    float m_debugNoticeSeconds = 0.0f;

    TextEntrySession m_textEntry;

    Layer1Row m_layer1Selection = Layer1Row::WorkspaceSelection;
    Layer2Row m_layer2Selection = Layer2Row::ParticleSpecies;
    bool m_densityEditingExponent = false;

    WorkspaceStatusTone m_statusTone = WorkspaceStatusTone::Ready;

    RuntimeSubLayerTraversal m_subLayers;
    bool m_displaySliders = false;
    bool m_initialized = false;
    bool m_active = false;
    bool m_paused = true;
    bool m_runtimeEnabled = false;

    unsigned int m_activeMarkerCount = 0;

    // Physical seconds, deliberately separate from wall/camera time.
    static constexpr double kPhysicalSecondsPerWallSecond = 1.0e-11;
    static constexpr double kMaxPhysicsStepSeconds = 1.0e-13;
    double m_elapsedSimulationTime = 0.0;
};

#endif
