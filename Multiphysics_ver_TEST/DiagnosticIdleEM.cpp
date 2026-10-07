#include "DiagnosticIdleEM.h"
#include "rendererEM_Euclid.h"
#include "TheArbiterEM.h"
#include "CameraEM.h"

#include <algorithm>
#include <cmath>
#include <string>

using namespace glm;

bool DiagnosticIdle::initialize(WorkspaceServices& services) {
    m_arbiter = services.arbiter;
    if (!services.renderer || !m_arbiter) return false;

    const auto* preset = findSimulationPreset(services.renderer->getSimBoxSize());
    if (!preset || !preset->enabled || preset->collisionGridDim != services.renderer->getGridDimSize()) return false;
    m_activePreset = *preset;
    m_requestedSimBoxSize = preset->boxSize;
    m_visualGridDim = preset->collisionGridDim;
    m_visualBoundarySize = m_visualGridSize = m_visualPlaneSize = static_cast<float>(m_activePreset.boxSize);
    refreshTransitionGrid(services);
    return true;
}

void DiagnosticIdle::refreshTransitionGrid(const WorkspaceServices& services) {
    m_transitionGridDimension = std::max(1, services.renderer->getGridDimSize());
    m_transitionGridMajorEvery = std::max(1, services.renderer->getGridMajorEvery());
    m_mulphyMajorCount = std::max(
        1,
        (m_transitionGridDimension + m_transitionGridMajorEvery - 1) /
        m_transitionGridMajorEvery);
}

void DiagnosticIdle::enter(WorkspaceServices& services) {
    (void)services;
}

void DiagnosticIdle::exit(WorkspaceServices& services) {
    (void)services;
}

void DiagnosticIdle::update(
    const WorkspaceFrameContext& frame,
    WorkspaceServices& services) {

    const float dt = std::isfinite(frame.deltaTime) ? std::max(0.0f, frame.deltaTime) : 0.0f;
    if (boxResizeActive()) {
        m_previewRotationDegrees += kPreviewRotationSpeed * dt;
        updateBoxResize(dt, services);
        return;
    }

    switch (m_visualTransition) {
    case VisualTransitionState::Idle:
        m_previewRotationDegrees += kPreviewRotationSpeed * dt;
        m_sliceTravel += kSliceCycleSpeed * dt;
        break;

    case VisualTransitionState::Mulphy_OrientToFront:
        m_previewRotationDegrees += kTransitionRotationSpeed * dt;
        m_sliceTravel += kTransitionSliceSpeed * dt;
        if (m_previewRotationDegrees >= m_targetRotationDegrees) {
            m_previewRotationDegrees = m_targetRotationDegrees;

            float target = std::floor(m_sliceTravel / 3.0f) * 3.0f;
            if (target <= m_sliceTravel) target += 3.0f;

            m_targetSliceTravel = target;
            m_visualTransition = VisualTransitionState::Mulphy_WaitForXYStart;
        }
        break;

    case VisualTransitionState::Mulphy_WaitForXYStart:
        m_sliceTravel += kTransitionSliceSpeed * dt;
        if (m_sliceTravel >= m_targetSliceTravel) {
            m_sliceTravel = m_targetSliceTravel;
            m_mulphyPlaneProgress = 0.0f;
            m_mulphyTargetMajorIndex = 1;
            m_visualTransition = VisualTransitionState::Mulphy_SweepToMajor;
        }
        break;

    case VisualTransitionState::Mulphy_SweepToMajor: {
        const float targetProgress = std::min(
            1.0f,
            static_cast<float>(
                std::min(
                    m_transitionGridDimension,
                    m_mulphyTargetMajorIndex * m_transitionGridMajorEvery)) /
            static_cast<float>(m_transitionGridDimension));

        m_mulphyPlaneProgress = std::min(
            targetProgress,
            m_mulphyPlaneProgress + kTransitionSliceSpeed * dt);
        m_sliceTravel = m_targetSliceTravel + m_mulphyPlaneProgress;

        if (m_mulphyPlaneProgress >= targetProgress) {
            m_mulphyClearedProgress = targetProgress;
            m_mulphyMajorHoldElapsed = 0.0f;
            m_visualTransition = VisualTransitionState::Mulphy_HoldMajor;
        }
        break;
    }

    case VisualTransitionState::Mulphy_HoldMajor:
        m_mulphyMajorHoldElapsed += dt;
        if (m_mulphyMajorHoldElapsed < kMulphyMajorHoldDuration) break;

        if (m_mulphyPlaneProgress >= 1.0f) {
            m_mulphyAxisProgress = 0.0f;
            m_visualTransition = VisualTransitionState::Mulphy_AxisToOrigin;
        }
        else {
            ++m_mulphyTargetMajorIndex;
            m_visualTransition = VisualTransitionState::Mulphy_SweepToMajor;
        }
        break;

    case VisualTransitionState::Mulphy_AxisToOrigin:
        m_mulphyAxisProgress = std::min(
            1.0f,
            m_mulphyAxisProgress + dt / kMulphyAxisTransitionDuration);
        if (m_mulphyAxisProgress >= 1.0f) {
            m_visualTransition = VisualTransitionState::Mulphy_HoldVolume;
            m_mulphyEnterComplete = true;
        }
        break;

    case VisualTransitionState::Mulphy_HoldVolume:
        break;

    case VisualTransitionState::Mulphy_ReturnAxisToCenter:
        m_mulphyAxisProgress = std::max(
            0.0f,
            m_mulphyAxisProgress - dt / kMulphyAxisTransitionDuration);
        if (m_mulphyAxisProgress <= 0.0f) {
            m_mulphyPlaneProgress = 1.0f;
            m_mulphyTargetMajorIndex = std::max(0, m_mulphyMajorCount - 1);
            m_visualTransition = VisualTransitionState::Mulphy_ReturnSweepToMajor;
        }
        break;

    case VisualTransitionState::Mulphy_ReturnSweepToMajor: {
        const int targetCell = std::min(
            m_transitionGridDimension,
            m_mulphyTargetMajorIndex * m_transitionGridMajorEvery);
        const float targetProgress =
            static_cast<float>(targetCell) /
            static_cast<float>(m_transitionGridDimension);

        m_mulphyPlaneProgress = std::max(
            targetProgress,
            m_mulphyPlaneProgress - kTransitionSliceSpeed * dt);
        m_sliceTravel = m_targetSliceTravel + m_mulphyPlaneProgress;

        if (m_mulphyPlaneProgress <= targetProgress) {
            m_mulphyClearedProgress = targetProgress;
            m_mulphyMajorHoldElapsed = 0.0f;
            m_visualTransition = VisualTransitionState::Mulphy_ReturnHoldMajor;
        }
        break;
    }

    case VisualTransitionState::Mulphy_ReturnHoldMajor:
        m_mulphyMajorHoldElapsed += dt;
        if (m_mulphyMajorHoldElapsed < kMulphyMajorHoldDuration) break;

        if (m_mulphyTargetMajorIndex <= 0) {
            m_mulphyPlaneProgress = 0.0f;
            m_sliceTravel = m_targetSliceTravel;
            m_visualTransition = VisualTransitionState::Idle;
            m_mulphyReturnComplete = true;
        }
        else {
            --m_mulphyTargetMajorIndex;
            m_visualTransition = VisualTransitionState::Mulphy_ReturnSweepToMajor;
        }
        break;
    }
}

void DiagnosticIdle::render(
    const WorkspaceFrameContext& frame,
    WorkspaceServices& services) {

    (void)frame;
    if (!services.renderer) return;

    EuclidRenderer& renderer = *services.renderer;
    EuclidRenderer::UniformGrid grid;

    const int gridDim = std::max(1, renderer.getGridDimSize());
    const int majorEvery = std::max(1, renderer.getGridMajorEvery());

    const float boxSize = static_cast<float>(renderer.getSimBoxSize());
    const float halfBox = boxSize * 0.5f;
    const float cellSize = boxSize / static_cast<float>(gridDim);

    grid.dimensions = ivec3(gridDim);
    grid.origin = vec3(-halfBox);
    grid.cellSize = vec3(cellSize);
    grid.majorEvery = majorEvery;

    EuclidRenderer::GridDisplay display;
    display.boundary = true;
    display.majorGrid = true;
    display.minorGrid = false;
    display.axes = true;

    glPushMatrix();

    const float visualRotation = std::fmod(m_previewRotationDegrees, 360.0f);
    glRotatef(visualRotation, 0.0f, 1.0f, 0.0f);

    const bool mulphySweepVisual =
        m_visualTransition == VisualTransitionState::Mulphy_SweepToMajor ||
        m_visualTransition == VisualTransitionState::Mulphy_HoldMajor ||
        m_visualTransition == VisualTransitionState::Mulphy_ReturnSweepToMajor ||
        m_visualTransition == VisualTransitionState::Mulphy_ReturnHoldMajor;

    const bool mulphyCleanVolumeVisual =
        m_visualTransition == VisualTransitionState::Mulphy_AxisToOrigin ||
        m_visualTransition == VisualTransitionState::Mulphy_HoldVolume ||
        m_visualTransition == VisualTransitionState::Mulphy_ReturnAxisToCenter;

    if (mulphySweepVisual || mulphyCleanVolumeVisual) {
        renderer.drawGridBoundary(grid);

        if (mulphySweepVisual) {
            EuclidRenderer::GridDisplay internalDisplay;
            internalDisplay.boundary = false;
            internalDisplay.majorGrid = true;
            internalDisplay.minorGrid = false;
            internalDisplay.axes = false;

            const float planePosition =
                grid.origin.z + boxSize * m_mulphyPlaneProgress;
            const float visibleGridStart =
                grid.origin.z + boxSize * m_mulphyClearedProgress;

            renderer.drawUniformGridZRange(
                grid,
                visibleGridStart,
                grid.origin.z + boxSize,
                internalDisplay);
            renderer.drawGridPlane(
                grid,
                EuclidRenderer::PLANE_XY,
                planePosition,
                false
            );
        }

        const vec3 axisOrigin = mix(
            vec3(0.0f),
            grid.origin,
            std::clamp(m_mulphyAxisProgress, 0.0f, 1.0f));
        renderer.drawAxisGizmo(axisOrigin, boxSize * 0.20f);

        glPopMatrix();
        return;
    }

    // Temporary extents never mutate the committed renderer/CUDA domain.
    const auto visualGrid = [&](float extent, int resolution) {
        EuclidRenderer::UniformGrid result = grid;
        result.dimensions = ivec3(resolution);
        result.origin = vec3(-extent * 0.5f);
        result.cellSize = vec3(extent / static_cast<float>(resolution));
        return result;
    };
    EuclidRenderer::UniformGrid planeGrid = grid;
    if (boxResizeActive()) {
        using Phase = BoxResizePhase;
        display.boundary = false;
        const auto newGrid = visualGrid(m_visualGridSize, m_visualGridDim);
        const auto oldGrid = visualGrid(static_cast<float>(m_resizeStep.from.boxSize), m_resizeStep.from.collisionGridDim);
        const float fromHalf = m_resizeStep.from.boxSize * 0.5f;
        const float toHalf = m_resizeStep.to.boxSize * 0.5f;
        const bool sweep = m_boxResizePhase == Phase::ExpandPlaneSweep || m_boxResizePhase == Phase::ShrinkPlaneSweep;
        const bool outerReveal = m_boxResizePhase == Phase::RevealOuterGrid;
        const bool reposition = m_boxResizePhase == Phase::RepositionOuterSlice;
        const bool reducingBoundary = m_boxResizePhase == Phase::ShrinkBoundary &&
            m_resizeStep.from.collisionGridDim != m_resizeStep.to.collisionGridDim;
        if (sweep) {
            display.axes = false;
            renderer.drawUniformGridZRange(oldGrid, m_visualPlanePosition, fromHalf, display);
            const float lowerZ = m_boxResizePhase == Phase::ExpandPlaneSweep ? -fromHalf : -m_visualGridSize * 0.5f;
            renderer.drawUniformGridZRange(newGrid, lowerZ, m_visualPlanePosition, display);
        }
        else if (reposition || outerReveal) {
            display.axes = false;
            // Retain the newly seeded central slabs while revealing the outer
            // volume from -Z to +Z. Disjoint ranges avoid double blending.
            if (outerReveal) renderer.drawUniformGridZRange(newGrid, -toHalf, m_visualPlanePosition, display);
            const float remainingStart = outerReveal ? std::max(-fromHalf, m_visualPlanePosition) : -fromHalf;
            renderer.drawUniformGridZRange(newGrid, remainingStart, fromHalf, display);
        }
        else renderer.drawUniformGrid(newGrid, display);
        renderer.drawGridBoundary(visualGrid(m_visualBoundarySize, m_visualGridDim));
        // Retain the old inner cube during expansion, and the destination
        // inner cube during shrink, until both boundaries coincide.
        if (m_visualBoundarySize > m_visualInnerSize)
            renderer.drawGridBoundary(visualGrid(m_visualInnerSize, m_visualGridDim));
        planeGrid = visualGrid(m_visualPlaneSize, m_visualGridDim);
        if (sweep || reposition || outerReveal || reducingBoundary) {
            renderer.drawAxisGizmo(vec3(0.0f), m_visualGridSize * 0.2f);
            renderer.drawGridPlane(planeGrid, EuclidRenderer::PLANE_XY, m_visualPlanePosition, false);
            glPopMatrix();
            return;
        }
    }
    else {
        renderer.drawUniformGrid(grid, display);
    }

    float sliceCycle = std::fmod(m_sliceTravel, 3.0f);
    if (sliceCycle < 0.0f) sliceCycle += 3.0f;

    const int segment = std::min(2, static_cast<int>(sliceCycle));
    const float local = sliceCycle - static_cast<float>(segment);

    const int halfSlice = gridDim / 2;
    const int sliceOffset = static_cast<int>(std::round(
        -halfSlice + local * static_cast<float>(halfSlice * 2)));
    const int sliceIndex = std::clamp(halfSlice + sliceOffset, 0, gridDim);

    EuclidRenderer::GridPlane plane = EuclidRenderer::PLANE_XY;
    float planePosition = 0.0f;

    switch (segment) {
    case 0:
        plane = EuclidRenderer::PLANE_XY;
        planePosition = planeGrid.origin.z +
            static_cast<float>(sliceIndex) * planeGrid.cellSize.z;
        break;
    case 1:
        plane = EuclidRenderer::PLANE_XZ;
        planePosition = planeGrid.origin.y +
            static_cast<float>(sliceIndex) * planeGrid.cellSize.y;
        break;
    default:
        plane = EuclidRenderer::PLANE_YZ;
        planePosition = planeGrid.origin.x +
            static_cast<float>(sliceIndex) * planeGrid.cellSize.x;
        break;
    }

    renderer.drawGridPlane(planeGrid, plane, planePosition, false);
    glPopMatrix();
}

bool DiagnosticIdle::handleInput(
    const WorkspaceInputEvent& input,
    WorkspaceServices& services) {

    if (!m_arbiter) m_arbiter = services.arbiter;
    if (!m_arbiter) return false;
    if (boxResizeActive()) return true;

    if (m_arbiter->isGlobalShell()) {
        switch (input.action) {
        case WorkspaceInputAction::Previous:
            moveGlobalShellCursor(-1);
            return true;
        case WorkspaceInputAction::Next:
            moveGlobalShellCursor(+1);
            return true;
        case WorkspaceInputAction::Decrease:
            adjustGlobalShellValue(-1);
            return true;
        case WorkspaceInputAction::Increase:
            adjustGlobalShellValue(+1);
            return true;
        case WorkspaceInputAction::Activate:
            if (m_activeShellRow != GlobalShellRow::Configure)
                return true;
            if (m_arbiter->getWorkspaceDomain() == TheArbiter::WorkspaceDomain::NONE) {
                beginBoxResize(m_requestedSimBoxSize, services);
                return true;
            }
            if (m_arbiter->getUnitMeasurement() ==
                TheArbiter::UnitMeasurement::IMPERIAL)
                return true;
            m_arbiter->requestEnterDomain(m_arbiter->getWorkspaceDomain());
            return true;

        case WorkspaceInputAction::Back:
            return true;
        default:
            return false;
        }
    }

    if (m_arbiter->isDomainSelection() &&
        input.action == WorkspaceInputAction::Back) {
        m_arbiter->requestReturnToGlobalShell(m_arbiter->getWorkspaceDomain());
        return true;
    }

    return false;
}

WorkspacePresentation DiagnosticIdle::buildPresentation() const {
    WorkspacePresentation p;
    p.panelVisible = true;
    p.workspaceName = "LAYER 0 -> GLOBAL SHELL CONFIG";

    const bool multiphysics = m_arbiter &&
        m_arbiter->getWorkspaceDomain() == TheArbiter::WorkspaceDomain::MULPHY_SIM;

    // ---------------------------------------------------------
    // Layer 0 menu rows
    // ---------------------------------------------------------
    WorkspacePanelSection section;
    section.rows.push_back({
        "[1]: DOMAIN SELECTION", 
        selectedEnvironmentName(), 
        true,
        m_activeShellRow == GlobalShellRow::Environment
    });

    section.rows.push_back({
        multiphysics 
        ? "[2]: SIMULATION MEASUREMENT" 
        : "[2]: SIMULATION BOX SIZE",
        multiphysics 
        ? selectedUnitMeasurementName() 
        : std::to_string(m_requestedSimBoxSize), 
        true,
        m_activeShellRow == GlobalShellRow::Configuration
    });

    section.rows.push_back({
        "[3]: E TO CONFIG GLOBAL SHELL", 
        "", 
        true,
        m_activeShellRow == GlobalShellRow::Configure
    });

    p.sections.push_back(section);

    // ---------------------------------------------------------
    // IDLE domain
    // ---------------------------------------------------------
    if (boxResizeActive()) {
        p.statusLine = m_resizeStep.to.boxSize > m_resizeStep.from.boxSize
            ? "AUTO: Increasing Simulation Box..." : "AUTO: Decreasing Simulation Box...";
        p.statusTone = p.frameTone = WorkspaceStatusTone::Transition;
        p.statusBlink = p.frameBlink = true;
    }
    else if (!multiphysics) {
        if (m_requestedSimBoxSize == m_activePreset.boxSize) {
            p.statusLine = "SIM SIZE READY: " + std::to_string(m_activePreset.boxSize) + " (" + std::to_string(m_activePreset.collisionGridDim) + "^3)";
            p.statusTone = WorkspaceStatusTone::Ready;
        }
        else if (isSupportedBoxSize(m_requestedSimBoxSize)) {
            p.statusLine = "SIM SIZE SELECT: " + std::to_string(m_requestedSimBoxSize) + " (" + std::to_string(findSimulationPreset(m_requestedSimBoxSize)->collisionGridDim) + "^3)";
            p.statusTone = WorkspaceStatusTone::Transition;
        }
        else {
            p.statusLine = "SIM SIZE UNAVAILABLE";
            p.statusTone = WorkspaceStatusTone::Warning;
        }
    }
    // ---------------------------------------------------------
    // MULTIPHYSICS domain
    // ---------------------------------------------------------
    else if (m_arbiter->getUnitMeasurement() == TheArbiter::UnitMeasurement::IMPERIAL) {

        p.statusLine = "WARNING: UNIT MEASUREMENT UNAVAILABLE";
        p.statusTone = WorkspaceStatusTone::Warning;
    }
    else {

        p.statusLine ="READY: GLOBAL SHELL CONFIGURATION VALID";
        p.statusTone = WorkspaceStatusTone::Ready;
    }

    // ---------------------------------------------------------
    // Controls
    // ---------------------------------------------------------
    p.footerLine1 =
        "W/S: Select row    A/D: Change value    E/ENTER: Configure";

    p.footerLine2 =
        "ESC: Exit";

    return p;
}

DiagnosticIdle::BoxResizeState DiagnosticIdle::boxResizeState() const {
    const auto* requested = findSimulationPreset(m_requestedSimBoxSize);
    return {m_boxResizePhase, m_requestedSimBoxSize, m_activePreset.boxSize,
        m_visualBoundarySize, m_visualPlaneSize, m_visualGridSize,
        m_sliceTravel, m_slicePausedForResize,
        requested ? requested->collisionGridDim : 0, m_activePreset.collisionGridDim, m_visualGridDim,
        m_resizeStep.from.boxSize, m_resizeStep.to.boxSize, m_resizeSteps.size(),
        m_visualInnerSize, m_visualPlanePosition, m_sweepProgress};
}

void DiagnosticIdle::beginBoxResize(int targetSize, WorkspaceServices& services) {
    if (boxResizeActive() || !services.renderer || !isSupportedBoxSize(targetSize) ||
        targetSize == m_activePreset.boxSize) return;
    m_resizeSteps.clear();
    const int first = simulationPresetIndex(m_activePreset.boxSize);
    const int last = simulationPresetIndex(targetSize);
    const int direction = last > first ? 1 : -1;
    for (int index = first; index != last; index += direction) {
        if (!kSimulationPresets[index + direction].enabled) { m_resizeSteps.clear(); return; }
        m_resizeSteps.push_back({kSimulationPresets[index], kSimulationPresets[index + direction]});
    }
    startNextResizeStep();
}

void DiagnosticIdle::startNextResizeStep() {
    m_resizeStep = m_resizeSteps.front();
    m_visualBoundarySize = m_visualGridSize = m_visualPlaneSize = static_cast<float>(m_resizeStep.from.boxSize);
    m_visualInnerSize = static_cast<float>(std::min(m_resizeStep.from.boxSize, m_resizeStep.to.boxSize));
    m_visualGridDim = m_resizeStep.from.collisionGridDim;
    m_visualPlanePosition = 0.0f;
    m_sweepProgress = 0.0f;
    m_slicePausedForResize = false;
    m_resizePhaseElapsed = 0.0f;
    m_boxResizePhase = BoxResizePhase::SetHold;
}

void DiagnosticIdle::waitForSliceStart(BoxResizePhase phase) {
    m_boxResizePhase = phase;
    m_resizePhaseElapsed = 0.0f;
    m_resizeSliceStart = (std::floor(m_sliceTravel / 3.0f) + 1.0f) * 3.0f;
}

void DiagnosticIdle::waitForSliceCenter(BoxResizePhase phase) {
    m_boxResizePhase = phase;
    m_resizePhaseElapsed = 0.0f;
    // Strictly future XY -Z -> 0 crossing, including requests made after
    // the current XY midpoint or during XZ/YZ. Clamp to it on crossing.
    m_resizeSliceCenter = (std::floor((m_sliceTravel - 0.5f) / 3.0f) + 1.0f) * 3.0f + 0.5f;
}

void DiagnosticIdle::updateBoxResize(float dt, WorkspaceServices& services) {
    using Phase = BoxResizePhase;
    if (services.camera) services.camera->updatePoseTransition(dt);
    m_resizePhaseElapsed += dt;
    const auto next = [&](Phase phase) {
        m_boxResizePhase = phase;
        m_resizePhaseElapsed = 0.0f;
    };
    const auto moveCamera = [&]() {
        if (services.camera) services.camera->beginRelativeDistanceScale(
            static_cast<float>(m_resizeStep.to.boxSize) / m_resizeStep.from.boxSize, kResizeStageDuration);
    };
    const auto animate = [&](float& extent) {
        const float t = std::clamp(m_resizePhaseElapsed / kResizeStageDuration, 0.0f, 1.0f);
        extent = mix(static_cast<float>(m_resizeStep.from.boxSize), static_cast<float>(m_resizeStep.to.boxSize),
            t * t * (3.0f - 2.0f * t));
        return t >= 1.0f;
    };
    const bool changesResolution = m_resizeStep.from.collisionGridDim != m_resizeStep.to.collisionGridDim;
    switch (m_boxResizePhase) {
    case Phase::SetHold:
        m_sliceTravel += kSliceCycleSpeed * dt;
        if (m_resizePhaseElapsed < kResizeHoldDuration) break;
        if (m_resizeStep.to.boxSize > m_resizeStep.from.boxSize) {
            next(Phase::ExpandBoundary);
            moveCamera();
        }
        else if (changesResolution) {
            waitForSliceStart(Phase::WaitForShrinkSliceStart);
            moveCamera();
        }
        else waitForSliceCenter(Phase::WaitForShrinkSliceCenter);
        break;
    case Phase::ExpandBoundary:
        m_sliceTravel += kSliceCycleSpeed * dt;
        if (animate(m_visualBoundarySize)) {
            if (changesResolution) waitForSliceStart(Phase::WaitForExpandSliceStart);
            else waitForSliceCenter(Phase::WaitForExpandSliceCenter);
        }
        break;
    case Phase::WaitForExpandSliceStart:
    case Phase::WaitForShrinkSliceStart: {
        const float previous = m_sliceTravel;
        m_sliceTravel += kSliceCycleSpeed * dt;
        if (previous < m_resizeSliceStart && m_sliceTravel >= m_resizeSliceStart) {
            m_sliceTravel = m_resizeSliceStart;
            m_visualPlanePosition = -m_resizeStep.from.boxSize * 0.5f;
            next(m_resizeStep.to.boxSize > m_resizeStep.from.boxSize ? Phase::ExpandPlaneSweep : Phase::ShrinkPlaneSweep);
        }
        break;
    }
    case Phase::ExpandPlaneSweep:
    case Phase::ShrinkPlaneSweep: {
        m_sweepProgress = std::clamp(m_resizePhaseElapsed / kResizeSweepDuration, 0.0f, 1.0f);
        const float eased = m_sweepProgress * m_sweepProgress * (3.0f - 2.0f * m_sweepProgress);
        m_visualPlaneSize = m_visualGridSize = mix(static_cast<float>(m_resizeStep.from.boxSize),
            static_cast<float>(m_resizeStep.to.boxSize), eased);
        // Intermediate resolutions are visual only; CUDA sees power-of-two presets at commit.
        m_visualGridDim = static_cast<int>(std::round(mix(static_cast<float>(m_resizeStep.from.collisionGridDim),
            static_cast<float>(m_resizeStep.to.collisionGridDim), eased)));
        m_visualPlanePosition = (m_sweepProgress - 0.5f) * m_resizeStep.from.boxSize;
        m_sliceTravel = m_resizeSliceStart + m_sweepProgress;
        if (m_sweepProgress >= 1.0f)
            next(m_boxResizePhase == Phase::ExpandPlaneSweep ? Phase::RepositionOuterSlice : Phase::ShrinkBoundary);
        break;
    }
    case Phase::RepositionOuterSlice: {
        const float t = std::clamp(m_resizePhaseElapsed / kSliceRepositionDuration, 0.0f, 1.0f);
        m_visualPlanePosition = mix(m_resizeStep.from.boxSize * 0.5f, -m_resizeStep.to.boxSize * 0.5f,
            t * t * (3.0f - 2.0f * t));
        if (t >= 1.0f) { m_sweepProgress = 0.0f; next(Phase::RevealOuterGrid); }
        break;
    }
    case Phase::RevealOuterGrid:
        m_sweepProgress = std::clamp(m_resizePhaseElapsed / kResizeSweepDuration, 0.0f, 1.0f);
        m_visualPlanePosition = (m_sweepProgress - 0.5f) * m_resizeStep.to.boxSize;
        if (m_sweepProgress >= 1.0f) completeBoxResize(services);
        break;
    case Phase::WaitForExpandSliceCenter:
    case Phase::WaitForShrinkSliceCenter: {
        const float previous = m_sliceTravel;
        m_sliceTravel += kSliceCycleSpeed * dt;
        if (previous < m_resizeSliceCenter && m_sliceTravel >= m_resizeSliceCenter) {
            m_sliceTravel = m_resizeSliceCenter;
            m_slicePausedForResize = true;
            next(m_resizeStep.to.boxSize > m_resizeStep.from.boxSize ? Phase::ExpandPlane : Phase::ShrinkGrid);
        }
        break;
    }
    case Phase::ExpandPlane:
        if (animate(m_visualPlaneSize)) next(Phase::ExpandGrid);
        break;
    case Phase::ExpandGrid:
        if (animate(m_visualGridSize)) completeBoxResize(services);
        break;
    case Phase::ShrinkGrid:
        if (animate(m_visualGridSize)) next(Phase::ShrinkPlane);
        break;
    case Phase::ShrinkPlane:
        if (animate(m_visualPlaneSize)) {
            next(Phase::ShrinkBoundary);
            moveCamera();
        }
        break;
    case Phase::ShrinkBoundary:
        if (changesResolution) {
            const float t = std::clamp(m_resizePhaseElapsed / kResizeStageDuration, 0.0f, 1.0f);
            m_visualPlanePosition = mix(m_resizeStep.from.boxSize * 0.5f, m_resizeStep.to.boxSize * 0.5f,
                t * t * (3.0f - 2.0f * t));
        }
        if (animate(m_visualBoundarySize)) completeBoxResize(services);
        break;
    case Phase::Idle:
        break;
    }
}

void DiagnosticIdle::completeBoxResize(WorkspaceServices& services) {

    services.renderer->setSimBoxSize(m_resizeStep.to.boxSize);
    services.renderer->setGridDimSize(m_resizeStep.to.collisionGridDim);
    services.renderer->setGridMajorEvery(m_resizeStep.to.majorEvery);

    m_activePreset = m_resizeStep.to;

    refreshTransitionGrid(services);

    m_visualGridDim = m_activePreset.collisionGridDim;

    m_visualBoundarySize = 
        m_visualGridSize = 
        m_visualPlaneSize = 
        static_cast<float>(m_activePreset.boxSize);

    m_slicePausedForResize = false;
    m_resizePhaseElapsed = 0.0f;
    m_resizeSteps.pop_front();

    if (m_resizeSteps.empty()) m_boxResizePhase = BoxResizePhase::Idle;
    else startNextResizeStep(); // Keep input locked and AUTO blinking across adjacent commits.
    // Preserve the slice phase: same-resolution steps resume at the XY center;
    // resolution-changing steps have completed XY and resume the normal cycle.
}

void DiagnosticIdle::beginMulphyEnterTransition() {
    m_mulphyEnterComplete = false;
    m_mulphyReturnComplete = false;
    m_mulphyPlaneProgress = 0.0f;
    m_mulphyClearedProgress = 0.0f;
    m_mulphyAxisProgress = 0.0f;
    m_mulphyMajorHoldElapsed = 0.0f;
    m_mulphyTargetMajorIndex = 1;

    const float revolution = std::floor(m_previewRotationDegrees / 360.0f);
    m_targetRotationDegrees = (revolution + 1.0f) * 360.0f;
    m_visualTransition = VisualTransitionState::Mulphy_OrientToFront;
}

void DiagnosticIdle::beginMulphyReturnTransition() {
    m_mulphyReturnComplete = false;
    m_mulphyAxisProgress = 1.0f;
    m_mulphyPlaneProgress = 1.0f;
    m_mulphyClearedProgress = 1.0f;
    m_mulphyMajorHoldElapsed = 0.0f;
    m_mulphyTargetMajorIndex = std::max(0, m_mulphyMajorCount - 1);
    m_visualTransition = VisualTransitionState::Mulphy_ReturnAxisToCenter;
}

const char* DiagnosticIdle::selectedEnvironmentName() const {
    if (!m_arbiter) return "IDLE";

    switch (m_arbiter->getWorkspaceDomain()) {
    case TheArbiter::WorkspaceDomain::MULPHY_SIM: return "MULTIPHYSICS";
    case TheArbiter::WorkspaceDomain::NONE:
    default:
        return "IDLE";
    }
}

const char* DiagnosticIdle::selectedUnitMeasurementName() const {
    if (!m_arbiter)
        return "METRIC";

    switch (m_arbiter->getUnitMeasurement()) {

    case TheArbiter::UnitMeasurement::IMPERIAL:
        return "IMPERIAL";

    case TheArbiter::UnitMeasurement::METRIC:
    default:
        return "METRIC";
    }
}

void DiagnosticIdle::cycleEnvironment(int direction) {
    if (!m_arbiter || direction == 0) return;

    using Domain = TheArbiter::WorkspaceDomain;
    static constexpr Domain order[] = {
        Domain::NONE,
        Domain::MULPHY_SIM
    };

    int current = 0;
    for (int i = 0; i < 2; i++) {
        if (order[i] == m_arbiter->getWorkspaceDomain()) {
            current = i;
            break;
        }
    }

    const int step = direction < 0 ? -1 : 1;
    const int next = (current + step + 2) % 2;
    m_arbiter->setWorkspaceDomain(order[next]);
}

void DiagnosticIdle::cycleUnitMeasurement(int direction) {
    if (!m_arbiter || direction == 0)
        return;

    using Unit = TheArbiter::UnitMeasurement;

    const Unit current =
        m_arbiter->getUnitMeasurement();

    if (current == Unit::METRIC) {
        m_arbiter->setUnitMeasurement(Unit::IMPERIAL);
    }
    else {
        m_arbiter->setUnitMeasurement(Unit::METRIC);
    }
}

void DiagnosticIdle::moveGlobalShellCursor(int direction) {
    if (direction == 0) return;

    const int count = static_cast<int>(GlobalShellRow::Count);
    const int current = static_cast<int>(m_activeShellRow);
    const int step = direction < 0 ? -1 : 1;

    m_activeShellRow = static_cast<GlobalShellRow>(
        (current + step + count) % count);
}

void DiagnosticIdle::adjustGlobalShellValue(int direction) {
    if (direction == 0) return;

    switch (m_activeShellRow) {
    case GlobalShellRow::Environment:
        cycleEnvironment(direction);
        break;

    case GlobalShellRow::Configuration: {
        if (m_arbiter && m_arbiter->getWorkspaceDomain() == TheArbiter::WorkspaceDomain::MULPHY_SIM) {
            cycleUnitMeasurement(direction);
            break;
        }
        static constexpr int kBoxSizes[] = { 2, 4, 8, 16, 32 };
        int index = 0;

        for (int i = 0; i < 5; i++) {
            if (kBoxSizes[i] == m_requestedSimBoxSize) {
                index = i;
                break;
            }
        }

        const int step = direction < 0 ? -1 : 1;
        index = (index + step + 5) % 5;
        m_requestedSimBoxSize = kBoxSizes[index];
        break;
    }

    case GlobalShellRow::Configure:
    case GlobalShellRow::Count:
    default:
        break;
    }
}
