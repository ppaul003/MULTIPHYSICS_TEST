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

    m_activeSimBoxSize = m_requestedSimBoxSize = services.renderer->getSimBoxSize();
    m_visualBoundarySize = m_visualGridSize = m_visualPlaneSize = static_cast<float>(m_activeSimBoxSize);

    m_transitionGridDimension = std::max(1, services.renderer->getGridDimSize());
    m_transitionGridMajorEvery = std::max(1, services.renderer->getGridMajorEvery());
    m_mulphyMajorCount = std::max(
        1,
        (m_transitionGridDimension + m_transitionGridMajorEvery - 1) /
        m_transitionGridMajorEvery);
    return true;
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
                false);
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
    const auto visualGrid = [&](float extent) {
        EuclidRenderer::UniformGrid result = grid;
        result.origin = vec3(-extent * 0.5f);
        result.cellSize = vec3(extent / static_cast<float>(gridDim));
        return result;
    };
    EuclidRenderer::UniformGrid planeGrid = grid;
    if (boxResizeActive()) {
        display.boundary = false;
        renderer.drawUniformGrid(visualGrid(m_visualGridSize), display);
        renderer.drawGridBoundary(visualGrid(m_visualBoundarySize));
        // Retain the old inner cube during expansion, and the destination
        // inner cube during shrink, until both boundaries coincide.
        const float innerSize = static_cast<float>(std::min(m_resizeFromSize, m_resizeToSize));
        if (m_visualBoundarySize > innerSize)
            renderer.drawGridBoundary(visualGrid(innerSize));
        planeGrid = visualGrid(m_visualPlaneSize);
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
        p.statusLine = m_resizeToSize > m_resizeFromSize
            ? "AUTO: Expanding Simulation Box..." : "AUTO: Shrinking Simulation Box...";
        p.statusTone = p.frameTone = WorkspaceStatusTone::Transition;
        p.statusBlink = p.frameBlink = true;
    }
    else if (!multiphysics) {
        if (m_requestedSimBoxSize == m_activeSimBoxSize) {
            p.statusLine = "SIM SIZE READY: " + std::to_string(m_activeSimBoxSize) + " (64^3)";
            p.statusTone = WorkspaceStatusTone::Ready;
        }
        else if (isSupportedBoxSize(m_requestedSimBoxSize)) {
            p.statusLine = "SIM SIZE SELECT: " + std::to_string(m_requestedSimBoxSize) + " (64^3)";
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
        p.postStatusLines.push_back("([2]: IMPERIAL selected)");
    }
    else {

        p.statusLine ="READY: GLOBAL SHELL CONFIGURATION VALID";
        p.statusTone = WorkspaceStatusTone::Ready;
        p.postStatusLines.push_back("([2]: METRIC selected)");
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
    return {m_boxResizePhase, m_requestedSimBoxSize, m_activeSimBoxSize,
        m_visualBoundarySize, m_visualPlaneSize, m_visualGridSize,
        m_sliceTravel, m_slicePausedForResize};
}

void DiagnosticIdle::beginBoxResize(int targetSize, WorkspaceServices& services) {
    if (boxResizeActive() || !services.renderer || !isSupportedBoxSize(targetSize) ||
        targetSize == m_activeSimBoxSize) return;
    m_resizeFromSize = m_activeSimBoxSize;
    m_resizeToSize = targetSize;
    m_visualBoundarySize = m_visualGridSize = m_visualPlaneSize = static_cast<float>(m_resizeFromSize);
    m_slicePausedForResize = false;
    m_resizePhaseElapsed = 0.0f;
    m_boxResizePhase = BoxResizePhase::SetHold;
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
            static_cast<float>(m_resizeToSize) / m_resizeFromSize, kResizeStageDuration);
    };
    const auto animate = [&](float& extent) {
        const float t = std::clamp(m_resizePhaseElapsed / kResizeStageDuration, 0.0f, 1.0f);
        extent = mix(static_cast<float>(m_resizeFromSize), static_cast<float>(m_resizeToSize),
            t * t * (3.0f - 2.0f * t));
        return t >= 1.0f;
    };
    switch (m_boxResizePhase) {
    case Phase::SetHold:
        m_sliceTravel += kSliceCycleSpeed * dt;
        if (m_resizePhaseElapsed < kResizeHoldDuration) break;
        if (m_resizeToSize > m_resizeFromSize) {
            next(Phase::ExpandBoundary);
            moveCamera();
        }
        else waitForSliceCenter(Phase::WaitForShrinkSliceCenter);
        break;
    case Phase::ExpandBoundary:
        m_sliceTravel += kSliceCycleSpeed * dt;
        if (animate(m_visualBoundarySize)) waitForSliceCenter(Phase::WaitForExpandSliceCenter);
        break;
    case Phase::WaitForExpandSliceCenter:
    case Phase::WaitForShrinkSliceCenter: {
        const float previous = m_sliceTravel;
        m_sliceTravel += kSliceCycleSpeed * dt;
        if (previous < m_resizeSliceCenter && m_sliceTravel >= m_resizeSliceCenter) {
            m_sliceTravel = m_resizeSliceCenter;
            m_slicePausedForResize = true;
            next(m_resizeToSize > m_resizeFromSize ? Phase::ExpandPlane : Phase::ShrinkGrid);
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
        if (animate(m_visualBoundarySize)) completeBoxResize(services);
        break;
    case Phase::Idle:
        break;
    }
}

void DiagnosticIdle::completeBoxResize(WorkspaceServices& services) {
    services.renderer->setSimBoxSize(m_resizeToSize);
    m_activeSimBoxSize = m_resizeToSize;
    m_visualBoundarySize = m_visualGridSize = m_visualPlaneSize = static_cast<float>(m_activeSimBoxSize);
    m_slicePausedForResize = false;
    m_resizePhaseElapsed = 0.0f;
    m_boxResizePhase = BoxResizePhase::Idle;
    // Keep m_sliceTravel exactly at the XY center. Normal update resumes +Z.
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

        for (int i = 0; i < 5; ++i) {
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
