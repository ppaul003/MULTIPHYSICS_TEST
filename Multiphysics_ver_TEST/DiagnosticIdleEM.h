#ifndef NDMSM_DIAGNOSTIC_IDLE_EM_H
#define NDMSM_DIAGNOSTIC_IDLE_EM_H

#include "IWorkspaceEM.h"
#include "SimulationPreset.h"
#include <deque>

class TheArbiter;

class DiagnosticIdle : public IWorkspace {
public:
    enum class BoxResizePhase {
        Idle, SetHold, ExpandBoundary, WaitForExpandSliceCenter,
        ExpandPlane, ExpandGrid, WaitForShrinkSliceCenter,
        ShrinkGrid, ShrinkPlane, ShrinkBoundary,
        WaitForExpandSliceStart, ExpandPlaneSweep, RepositionOuterSlice,
        RevealOuterGrid, WaitForShrinkSliceStart, ShrinkPlaneSweep
    };

    // Read-only snapshot for integration diagnostics; no mutable internals escape.
    struct BoxResizeState {
        BoxResizePhase phase;
        int requestedSize, activeSize;
        float boundarySize, planeSize, gridSize, sliceTravel;
        bool slicePaused;
        int requestedGridDim, activeGridDim, visualGridDim;
        int stepFromSize, stepToSize;
        std::size_t remainingSteps;
        float innerSize, planePosition, sweepProgress;
    };
    bool boxResizeActive() const { return m_boxResizePhase != BoxResizePhase::Idle; }
    BoxResizeState boxResizeState() const;
    enum class VisualTransitionState {
        Idle = 0,

        Mulphy_OrientToFront,
        Mulphy_WaitForXYStart,
        Mulphy_SweepToMajor,
        Mulphy_HoldMajor,
        Mulphy_AxisToOrigin,
        Mulphy_HoldVolume,
        Mulphy_ReturnAxisToCenter,
        Mulphy_ReturnSweepToMajor,
        Mulphy_ReturnHoldMajor
    };

    bool initialize(WorkspaceServices& services) override;
    void enter(WorkspaceServices& services) override;
    void exit(WorkspaceServices& services) override;

    void update(
        const WorkspaceFrameContext& frame,
        WorkspaceServices& services) override;

    void render(
        const WorkspaceFrameContext& frame,
        WorkspaceServices& services) override;

    bool handleInput(
        const WorkspaceInputEvent& input,
        WorkspaceServices& services) override;

    WorkspacePresentation buildPresentation() const override;

    void beginMulphyEnterTransition();
    void beginMulphyReturnTransition();

    bool mulphyEnterVisualComplete() const { return m_mulphyEnterComplete; }
    bool mulphyReturnVisualComplete() const { return m_mulphyReturnComplete; }

private:
    enum class GlobalShellRow {
        Environment = 0,
        Configuration,
        Configure,
        Count
    };

    const char* selectedEnvironmentName() const;
    const char* selectedUnitMeasurementName() const;
    void cycleEnvironment(int direction);
    void cycleUnitMeasurement(int direction);
    void moveGlobalShellCursor(int direction);
    void adjustGlobalShellValue(int direction);
    static bool isSupportedBoxSize(int size) { return supportedSimulationPreset(size); }
    struct TransitionStep { SimulationPreset from; SimulationPreset to; };
    void startNextResizeStep();
    void beginBoxResize(int targetSize, WorkspaceServices& services);
    void updateBoxResize(float dt, WorkspaceServices& services);
    void completeBoxResize(WorkspaceServices& services);
    void waitForSliceCenter(BoxResizePhase phase);
    void waitForSliceStart(BoxResizePhase phase);
    void refreshTransitionGrid(const WorkspaceServices& services);

private:
    static constexpr float kPreviewRotationSpeed = 25.0f;
    static constexpr float kSliceCycleSpeed = 0.35f;
    static constexpr float kTransitionRotationSpeed = 120.0f;
    static constexpr float kTransitionSliceSpeed = 1.40f;
    static constexpr float kMulphyMajorHoldDuration = 0.12f;
    static constexpr float kMulphyAxisTransitionDuration = 0.45f;

    TheArbiter* m_arbiter = nullptr;

    GlobalShellRow m_activeShellRow = GlobalShellRow::Environment;
    VisualTransitionState m_visualTransition = VisualTransitionState::Idle;
    
    bool m_mulphyEnterComplete = false;
    bool m_mulphyReturnComplete = false;

    float m_previewRotationDegrees = 0.0f;
    float m_sliceTravel = 0.0f;
    float m_targetRotationDegrees = 0.0f;
    float m_targetSliceTravel = 0.0f;
    float m_mulphyPlaneProgress = 0.0f;
    float m_mulphyClearedProgress = 0.0f;
    float m_mulphyAxisProgress = 0.0f;
    float m_mulphyMajorHoldElapsed = 0.0f;

    int m_requestedSimBoxSize = 4;
    SimulationPreset m_activePreset = kSimulationPresets[0];
    TransitionStep m_resizeStep{kSimulationPresets[0], kSimulationPresets[0]};
    std::deque<TransitionStep> m_resizeSteps;
    BoxResizePhase m_boxResizePhase = BoxResizePhase::Idle;
    float m_visualBoundarySize = 4.0f;
    float m_visualGridSize = 4.0f;
    float m_visualPlaneSize = 4.0f;
    int m_visualGridDim = 64;
    float m_visualInnerSize = 4.0f;
    float m_visualPlanePosition = 0.0f;
    float m_resizeSliceStart = 0.0f;
    float m_sweepProgress = 0.0f;
    float m_resizePhaseElapsed = 0.0f;
    float m_resizeSliceCenter = 0.5f;
    bool m_slicePausedForResize = false;

    static constexpr float kResizeHoldDuration = 0.20f;
    static constexpr float kResizeStageDuration = 0.50f;
    static constexpr float kResizeSweepDuration = 1.50f;
    static constexpr float kSliceRepositionDuration = 0.25f;

    int m_transitionGridDimension = 64;
    int m_transitionGridMajorEvery = 8;
    int m_mulphyMajorCount = 8;
    int m_mulphyTargetMajorIndex = 1;
};

#endif
