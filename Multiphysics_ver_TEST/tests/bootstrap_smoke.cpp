// Integration smoke test: real OpenGL/CUDA, public workspace APIs, no UI injection.
#include "TheTesseractEM.h"
#include "rendererEM_Euclid.h"
#include "CameraEM.h"
#include "ViewPortEM.h"
#include "EuclidEngineEM.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <array>
#include <vector>
#include <cmath>

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// Test-only read access to existing protected storage; no production API changes.
class CollisionTableProbe final : public ParticleSystem {
public:
    using ParticleSystem::ParticleSystem;
    std::array<const uint*, 4> cellTables() const {
        return {m_hCellStart, m_hCellEnd, m_dCellStart, m_dCellEnd};
    }
    float boundary() const { return m_params.boundary; }
};

int main(int argc, char** argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "--engine-lifecycle") {
            EuclidEngine engine;
            require(engine.init(argc, argv), "Engine initialization");
            glutHideWindow();
            glutTimerFunc(250, [](int) { glutLeaveMainLoop(); }, 0);
            engine.run();
            engine.shutdown();
            std::puts("PASS: engine main loop and shutdown (timer-driven, not an ESC keypress)");
            return 0;
        }
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
        glutInitWindowSize(1280, 900);
        const int window = glutCreateWindow("TEST bootstrap integration smoke");
        glutHideWindow();
        glewExperimental = GL_TRUE;
        require(glewInit() == GLEW_OK, "GLEW initialization");
        while (glGetError() != GL_NO_ERROR) {}
        std::printf("OpenGL: %s\n", glGetString(GL_VERSION));
        {
            CollisionTableProbe particles(16, make_uint3(64, 64, 64), true);
            particles.setActiveParticleCount(2);
            particles.setSimulationDomain(8.0f);
            particles.setUniformActiveRadii(0.0063f);
            const auto positionBuffer = particles.getCurrentReadBuffer();
            const auto radiiBuffer = particles.getRadiiBuffer();
            const float radius = particles.getParticleRadius();
            for (int box : {16, 32, 16, 8, 16}) {
                const uint dim = static_cast<uint>(findSimulationPreset(box)->collisionGridDim);
                const auto previousTables = particles.cellTables();
                const auto previousGrid = particles.getGridSize();
                particles.setSimulationDomain(static_cast<float>(box), make_uint3(dim, dim, dim));
                if (previousGrid.x == dim && previousGrid.y == dim && previousGrid.z == dim)
                    require(particles.cellTables() == previousTables, "Same-resolution resize preserves all host/device collision table addresses");
                require(particles.boundary() == box * 0.5f, "Collision boundary follows box size including +/-16");
                require(particles.getCapacity() == 16 && particles.getCurrentReadBuffer() == positionBuffer &&
                    particles.getRadiiBuffer() == radiiBuffer, "Grid resize preserves particle allocations");
                require(particles.getNumGridCells() == dim * dim * dim, "Cell tables resize with hash domain");
                const auto origin = particles.getWorldOrigin();
                const auto cell = particles.getCellSize();
                require(origin.x == -box * 0.5f && origin.y == origin.x && origin.z == origin.x &&
                    cell.x == box / float(dim) && cell.y == cell.x && cell.z == cell.x,
                    "GPU runtime domain follows box/grid preset including 32/128");
                require(particles.getParticleRadius() == radius, "Grid reallocation preserves collision radius");
                float radii[2]{};
                particles.dumpRadii(radii, 2);
                require(radii[0] == 0.0063f && radii[1] == 0.0063f, "Actual GPU radius data survives resize");
                const float half = box * 0.5f;
                require(particles.resetInBounds(ParticleSystem::CNFG_RANDOM_RESTART,
                    make_float3(half - 0.8f, half - 0.8f, half - 0.8f),
                    make_float3(half - 0.2f, half - 0.2f, half - 0.2f), 0.0063f), "Spawn near maximum hash indices");
                particles.update(1.0f / 60.0f);
                require(cudaDeviceSynchronize() == cudaSuccess, "Collision update accesses resized high-index cell tables");
                const auto position = particles.getSingleParticle(ParticleSystem::POSITION, 0);
                require(std::isfinite(position.x) && position.x <= half, "Particle remains valid inside resized boundary");
            }
            bool rejected = false;
            try { particles.setSimulationDomain(16.0f, make_uint3(96, 96, 96)); }
            catch (const std::invalid_argument&) { rejected = true; }
            require(rejected && particles.getGridSize().x == 128, "Non-power-of-two grid rejected without mutation");
            std::puts("PASS: CUDA cell reallocation, 16/32 table reuse, high-index collision access, VBO/radius/capacity preservation");
        }
        {
            // Destruction order keeps the GL context and renderer alive for CUDA cleanup.
            EuclidRenderer renderer;
            renderer.setWindowSize(1280, 900);
            renderer.setFOV(60.0f);
            renderer.setSimBoxSize(4);
            renderer.setGridDimSize(64);
            renderer.setGridMajorEvery(8);
            TheArbiter arbiter;
            CameraProcessor camera;
            camera.setBehaviorMode(CameraProcessor::CAM_MENU_PREVIEW);
            ViewPort viewport;
            viewport.resize(1280, 900);
            Tesseract host;
            WorkspaceServices services{&renderer, &arbiter, &viewport, &camera};
            require(host.initialize(services), "Workspace/CUDA initialization");
            require(cudaDeviceSynchronize() == cudaSuccess, "CUDA initialization synchronization");
            KeyboardInput keyboard;
            WorkspaceFrameContext frame;
            frame.deltaTime = 1.0f / 60.0f;
            frame.viewportWidth = 1280;
            frame.viewportHeight = 900;
            auto tick = [&]() {
                frame.elapsedTime += frame.deltaTime;
                host.update(frame);
                viewport.applyPerspective();
                glMatrixMode(GL_MODELVIEW);
                glLoadIdentity();
                camera.updateLag();
                camera.applyCameraTransform();
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                host.render(frame);
                viewport.drawOverlay(host.presentation());
                host.renderOverlay(frame);
                glFinish();
                require(glGetError() == GL_NO_ERROR, "OpenGL render error");
            };
            auto key = [&](unsigned char raw) {
                auto routed = arbiter.routeKeyboard(keyboard.onKey(raw, 0, 0));
                require(routed.hasWorkspaceInput, "Keyboard routing");
                host.handleInput(routed.workspaceInput);
                host.handleInputRelease(routed.workspaceInput);
            };
            auto settle = [&]() {
                for (int i = 0; i < 2400 && host.domainTransitionActive(); ++i) {
                    // Exercise every transition update; render periodically for speed.
                    frame.elapsedTime += frame.deltaTime;
                    host.update(frame);
                    if (i % 60 == 0) tick();
                }
                require(!host.domainTransitionActive(), "Transition must complete");
                tick();
            };
            auto value = [&](int row) { return host.presentation().sections.at(0).rows.at(row).value; };
            auto selectRow = [&](const std::string& label) {
                for (int attempt = 0; attempt < 16; ++attempt) {
                    for (const auto& section : host.presentation().sections)
                        for (const auto& row : section.rows)
                            if (row.selected && row.label.find(label) != std::string::npos) return;
                    key('s');
                }
                throw std::runtime_error("Missing menu row: " + label);
            };
            auto checkDomain = [&](int box) {
                const int grid = findSimulationPreset(box)->collisionGridDim;
                for (const auto& state : {host.particleDomainState(), host.atomicDomainState()}) {
                    require(state.collisionDimensions == glm::ivec3(grid), "CUDA dimensions follow active preset");
                    require(state.collisionCellCount == static_cast<unsigned int>(grid * grid * grid), "Collision cell table count follows preset");
                    require(state.collisionOrigin == glm::vec3(-box * 0.5f), "CUDA origin follows committed box");
                    require(state.collisionCellSize == glm::vec3(box / static_cast<float>(grid)), "CUDA collision spacing follows committed box");
                    require(state.physicalGrid.dimensions == glm::ivec3(8), "Physical grid remains 8^3");
                    require(state.physicalGrid.origin == glm::vec3(-box * 0.5f), "Physical origin follows committed box");
                    require(state.physicalGrid.voxelEdgeM == box / 8.0f, "Physical voxel spacing follows committed box");
                    SpawnDensityRegionGrid3D spawnGrid;
                    SpawnDensityRegion3D region;
                    require(spawnGrid.centeredRegion(state.physicalGrid, region), "Center spawn region remains valid");
                    require(region.volumeM3 == std::pow(box / 4.0f, 3.0f), "2x2x2 spawn volume follows physical geometry");
                    require(spawnGrid.regionCount(state.physicalGrid) == 64 &&
                        spawnGrid.selectionCount(state.physicalGrid) == 66, "Spawn selector includes whole domain, center and 64 physical regions");
                }
                const auto fields = host.atomicDomainState();
                const int fieldDimension = box <= 8 ? 8 : 16;
                require(fields.fieldGrid.dimensions == glm::ivec3(fieldDimension), "Independent field grid follows 8/16 sampling preset");
                require(fields.fieldGrid.origin == glm::vec3(-box * 0.5f) &&
                    fields.fieldGrid.voxelEdgeM == box / float(fieldDimension), "Field sampling geometry covers exact committed domain");
                require(fields.fieldGeometryValid && fields.fieldCellCount == fieldDimension * fieldDimension * fieldDimension,
                    "All seven fields borrow valid independent geometry (512/4096 cells)");
            };
            tick();
            require(arbiter.isGlobalShell() && value(0) == "IDLE", "Startup IDLE");
            require(host.presentation().sections.at(0).rows.size() == 3, "Exactly three shell rows");
            require(host.presentation().sections.at(0).rows.at(0).selected, "Row 1 remains default selected");
            require(host.boxResizeState().requestedSize == 4 && host.boxResizeState().activeSize == 4 && renderer.getSimBoxSize() == 4,
                "Startup requested/active/renderer size 4");
            require(host.presentation().statusLine == "SIM SIZE READY: 4 (64^3)" &&
                host.presentation().statusTone == WorkspaceStatusTone::Ready, "Startup ready status");
            checkDomain(4);
            selectRow("CONFIG GLOBAL SHELL"); key('e');
            require(!host.boxResizeActive() && !host.domainTransitionActive(), "Same-size commit is a no-op");

            auto resize = [&](int target, unsigned char commitKey) {
                using Phase = DiagnosticIdle::BoxResizePhase;
                const int from = renderer.getSimBoxSize();
                const int targetGrid = findSimulationPreset(target)->collisionGridDim;
                const float particleRadius = host.particleDomainState().collisionRadius;
                const float atomicRadius = host.atomicDomainState().collisionRadius;
                selectRow("SIMULATION BOX SIZE");
                for (int i = 0; i < 5 && value(1) != std::to_string(target); ++i) key('d');
                const std::string presetText = std::to_string(target) + " (" + std::to_string(targetGrid) + "^3)";
                require(host.presentation().statusLine == "SIM SIZE SELECT: " + presetText &&
                    host.presentation().statusTone == WorkspaceStatusTone::Transition, "Pending preset orange");
                require(renderer.getSimBoxSize() == from, "Selection does not commit");
                checkDomain(from);
                glm::vec3 eyeBefore, directionBefore;
                camera.getCenterViewRay(eyeBefore, directionBefore);
                selectRow("CONFIG GLOBAL SHELL"); key(commitKey);
                require(host.boxResizeActive() && !host.domainTransitionActive(), "Dedicated preset transaction starts");
                std::vector<int> commits;
                std::vector<Phase> phases;
                int stepFrom = from;
                int intermediateGridFrames = 0;
                float centerWaitStart = 0.0f;
                float pausedSlice = 0.0f;
                bool boundaryInterpolates = false, planeInterpolates = false, gridInterpolates = false;
                auto previous = host.boxResizeState();
                for (int i = 0; i < 5400 && host.boxResizeActive(); ++i) {
                    const auto state = host.boxResizeState();
                    if (phases.empty() || phases.back() != state.phase) {
                        phases.push_back(state.phase);
                        if (state.phase == Phase::WaitForExpandSliceCenter || state.phase == Phase::WaitForShrinkSliceCenter)
                            centerWaitStart = state.sliceTravel;
                    }
                    const auto presentation = host.presentation();
                    require(presentation.statusBlink && presentation.frameBlink &&
                        presentation.statusTone == WorkspaceStatusTone::Transition &&
                        presentation.frameTone == WorkspaceStatusTone::Transition, "Whole transaction blinks orange");
                    require(presentation.statusLine == (target > from ? "AUTO: Increasing Simulation Box..." :
                        "AUTO: Decreasing Simulation Box..."), "AUTO direction text");
                    require(renderer.getSimBoxSize() == state.activeSize && renderer.getGridDimSize() == state.activeGridDim &&
                        state.activeSize == state.stepFromSize, "Active preset remains step source until commit");
                    require(state.stepToSize == state.stepFromSize * 2 || state.stepFromSize == state.stepToSize * 2,
                        "Every step is adjacent");
                    require(state.requestedSize == target && state.requestedGridDim == targetGrid, "Final request survives intermediate steps");
                    require(state.remainingSteps > 0, "Active transaction has queued step");
                    if (i % 30 == 0) {
                        for (unsigned char raw : std::string("wsadeq\r")) key(raw);
                        require(value(1) == std::to_string(target) && value(0) == "IDLE" &&
                            host.presentation().sections.at(0).rows.at(2).selected, "Navigation locked across entire chain");
                        require(arbiter.routeKeyboard(keyboard.onKey(27, 0, 0)).arbiterCommand == TheArbiter::ArbiterCommand::CMD_EXIT,
                            "ESC still routed during transition");
                        checkDomain(state.activeSize);
                    }
                    const int sourceGrid = findSimulationPreset(state.stepFromSize)->collisionGridDim;
                    const int destinationGrid = findSimulationPreset(state.stepToSize)->collisionGridDim;
                    const bool changesGrid = sourceGrid != destinationGrid;
                    if (!changesGrid) {
                        require(state.visualGridDim == sourceGrid, "Same-resolution steps retain grid resolution");
                        require(state.phase != Phase::ExpandPlaneSweep && state.phase != Phase::ShrinkPlaneSweep &&
                            state.phase != Phase::RevealOuterGrid, "Same-resolution steps never use resolution-changing sweeps");
                    }
                    if (state.slicePaused) {
                        require(!changesGrid && std::fabs(std::fmod(state.sliceTravel, 3.0f) - 0.5f) < 1e-5f,
                            "Same-resolution steps retain center pause");
                        require(state.sliceTravel > centerWaitStart, "XY center crossing is strictly future");
                        if (previous.slicePaused) require(state.sliceTravel == pausedSlice, "Slice remains frozen throughout extent changes");
                        pausedSlice = state.sliceTravel;
                    }
                    if (!changesGrid && (state.phase == Phase::ExpandPlane || state.phase == Phase::ExpandGrid ||
                        state.phase == Phase::ShrinkGrid || state.phase == Phase::ShrinkPlane || state.phase == Phase::ShrinkBoundary))
                        require(state.slicePaused, "Same-resolution extent changes occur only while paused at center");
                    if (state.phase == Phase::ExpandPlaneSweep || state.phase == Phase::ShrinkPlaneSweep) {
                        require(!state.slicePaused && changesGrid, "Resolution change uses moving XY sweep");
                        require(std::fabs(state.planePosition - (state.sweepProgress - 0.5f) * state.stepFromSize) < 1e-5f,
                            "Plane position follows inner/outer Z pass");
                        if (state.visualGridDim > 64 && state.visualGridDim < 128) ++intermediateGridFrames;
                    }
                    if (state.phase == Phase::RevealOuterGrid) {
                        require(state.visualGridDim == destinationGrid && state.gridSize == state.stepToSize,
                            "Outer reveal uses destination grid");
                        require(std::fabs(state.planePosition - (state.sweepProgress - 0.5f) * state.stepToSize) < 1e-5f,
                            "Outer reveal traverses new -Z to +Z volume");
                    }
                    const float low = static_cast<float>(std::min(state.stepFromSize, state.stepToSize));
                    const float high = static_cast<float>(std::max(state.stepFromSize, state.stepToSize));
                    require(state.innerSize == low, "Nested inner boundary is explicit");
                    const auto between = [&](float extent) { return extent > low && extent < high; };
                    boundaryInterpolates |= between(state.boundarySize);
                    planeInterpolates |= between(state.planeSize);
                    gridInterpolates |= between(state.gridSize);
                    if (state.stepFromSize == previous.stepFromSize) {
                        require(std::fabs(state.boundarySize - previous.boundarySize) < high * 0.025f &&
                            std::fabs(state.planeSize - previous.planeSize) < high * 0.025f &&
                            std::fabs(state.gridSize - previous.gridSize) < high * 0.025f, "Continuous extents");
                    }
                    previous = state;
                    tick();
                    const int committed = renderer.getSimBoxSize();
                    if (committed != stepFrom) {
                        // Assert each adjacent step's complete identity, even inside a mixed chain.
                        std::vector<Phase> expectedPhases;
                        if (!changesGrid && committed > stepFrom)
                            expectedPhases = {Phase::SetHold, Phase::ExpandBoundary, Phase::WaitForExpandSliceCenter,
                                Phase::ExpandPlane, Phase::ExpandGrid};
                        else if (!changesGrid)
                            expectedPhases = {Phase::SetHold, Phase::WaitForShrinkSliceCenter,
                                Phase::ShrinkGrid, Phase::ShrinkPlane, Phase::ShrinkBoundary};
                        else if (committed > stepFrom)
                            expectedPhases = {Phase::SetHold, Phase::ExpandBoundary, Phase::WaitForExpandSliceStart,
                                Phase::ExpandPlaneSweep, Phase::RepositionOuterSlice, Phase::RevealOuterGrid};
                        else
                            expectedPhases = {Phase::SetHold, Phase::WaitForShrinkSliceStart,
                                Phase::ShrinkPlaneSweep, Phase::ShrinkBoundary};
                        require(phases == expectedPhases, "Adjacent step preserves the correct ordered animation path");
                        phases.clear();
                        commits.push_back(committed);
                        stepFrom = committed;
                        checkDomain(committed); // Includes intermediate commits while lock remains active.
                        if (committed != target)
                            require(host.boxResizeActive() && host.presentation().statusBlink, "No unlocked gap between steps");
                    }
                }
                require(!host.boxResizeActive(), "Transaction completes in bounded time");
                std::vector<int> expectedCommits;
                for (int size = from; size != target;) { size = target > from ? size * 2 : size / 2; expectedCommits.push_back(size); }
                require(commits == expectedCommits, "Commits follow adjacent ladder exactly");
                require(boundaryInterpolates && planeInterpolates && gridInterpolates, "All extents animate");
                const bool crossesResolutionChange = std::min(from, target) <= 8 && std::max(from, target) >= 16;
                if (crossesResolutionChange) {
                    require(intermediateGridFrames > 5, "Visible grid resolution changes progressively");
                }
                else require(intermediateGridFrames == 0, "Same-resolution transaction has no progressive resolution reveal");
                require(renderer.getSimBoxSize() == target && renderer.getGridDimSize() == targetGrid, "Destination box/grid committed");
                require(host.presentation().statusLine == "SIM SIZE READY: " + presetText &&
                    host.presentation().statusTone == WorkspaceStatusTone::Ready && !host.presentation().statusBlink &&
                    !host.presentation().frameBlink, "Ready only at final destination");
                require(host.boxResizeState().remainingSteps == 0, "Queue drained");
                checkDomain(target);
                require(host.particleDomainState().collisionRadius == particleRadius &&
                    host.atomicDomainState().collisionRadius == atomicRadius, "Preset transition preserves configured radii");
                glm::vec3 eyeAfter, directionAfter;
                camera.getCenterViewRay(eyeAfter, directionAfter);
                require(glm::length(directionBefore - directionAfter) < 1e-5f &&
                    glm::length(eyeAfter - eyeBefore * (float(target) / from)) < 1e-4f, "Chained camera scaling preserves angle");
                const float slice = host.boxResizeState().sliceTravel;
                tick();
                require(host.boxResizeState().sliceTravel > slice && !host.boxResizeState().slicePaused, "Idle slicing resumes");
                key(commitKey); require(!host.boxResizeActive(), "Same-preset commit is no-op");
                std::printf("PASS: preset %d -> %d, ordered phases, commits, input lock, camera and geometry\n", from, target);
            };

            // Start after the first XY midpoint: the wait must seek a future crossing.
            for (int i = 0; i < 130; ++i) host.update(frame);
            resize(8, 'e');
            require(host.boxResizeState().sliceTravel > 3.5f, "Wait skips already-passed XY center");

            auto unsupported = [&]() {
                const int active = renderer.getSimBoxSize();
                for (int unavailable : {2}) {
                    selectRow("SIMULATION BOX SIZE");
                    for (int i = 0; i < 5 && value(1) != std::to_string(unavailable); ++i) key('d');
                    require(host.presentation().statusTone == WorkspaceStatusTone::Warning, "Unsupported size warning");
                    selectRow("CONFIG GLOBAL SHELL"); key('e'); key(13);
                    require(!host.boxResizeActive() && !host.domainTransitionActive() && renderer.getSimBoxSize() == active,
                        "Unsupported presets never mutate committed geometry");
                    checkDomain(active);
                }
            };
            resize(16, 'e');
            unsupported();
            selectRow("DOMAIN SELECTION"); key('d');
            require(value(0) == "MULTIPHYSICS" && value(1) == "METRIC", "MULTIPHYSICS metric shell");
            key('d'); require(value(0) == "IDLE", "Domain cycle wraps");
            key('a'); require(value(0) == "MULTIPHYSICS", "Reverse domain cycle");
            selectRow("SIMULATION MEASUREMENT"); key('d');
            require(value(1) == "IMPERIAL" && host.presentation().statusTone == WorkspaceStatusTone::Warning, "Imperial warning");
            selectRow("CONFIG GLOBAL SHELL"); key('e'); key(13);
            require(arbiter.isGlobalShell() && !host.domainTransitionActive(), "Imperial blocks E and Enter");
            selectRow("SIMULATION MEASUREMENT"); key('a');
            selectRow("CONFIG GLOBAL SHELL"); key('e'); settle();
            require(arbiter.isDomainSelection(), "Metric enters MULTIPHYSICS despite unsupported IDLE draft");
            checkDomain(16);
            require(renderer.getSimBoxSize() == 16, "Domain entry preserves committed 16/128 preset");

            auto runAtomic = [&]() {
                // Domain entry starts on PARTICLE_SIM; select the new Atomic cartridge explicitly.
                require(arbiter.isDomainSelection(), "Atomic entry starts from Layer 1");
                selectRow("MULPHY_SIM SELECTION");
                for (int i = 0; i < 3 && arbiter.getActiveWorkspace() != TheArbiter::WorkspaceId::ATOMIC_PARTICLES; ++i)
                    key('d');
                require(arbiter.getActiveWorkspace() == TheArbiter::WorkspaceId::ATOMIC_PARTICLES, "Atomic workspace reachable");
                require(host.menu().items.empty(), "Atomic runtime menu absent outside Layer 3");
                selectRow("CONFIGURE WORKSPACE"); key('e');
                require(arbiter.getApplicationLayer() == TheArbiter::ApplicationLayer::WORKSPACE_CONFIGURATION, "Configure Atomic");
                selectRow("SELECT VOXEL SPAWN");
                require(value(6) == "[" + std::to_string(renderer.getSimBoxSize()) + " MICRO METER]^3",
                    "Whole-domain spawn remains default at every field resolution");
                key('d');
                require(value(6) == "VOXEL_CENTER", "Centered spawn remains the next option");
                key('d');
                for (int id = 0; id < 64; ++id) {
                    char expected[16];
                    std::snprintf(expected, sizeof(expected), "VOXEL_%03d", id);
                    require(value(6) == expected, "Existing voxel selection ordering remains unchanged");
                    key('d');
                }
                require(value(6) == "[" + std::to_string(renderer.getSimBoxSize()) + " MICRO METER]^3",
                    "Whole-domain option follows the 64 physical regions");
                key('d');
                require(value(6) == "VOXEL_CENTER", "Whole-domain option wraps back to center");
                selectRow("TOTAL GAS DENSITY"); key('e');
                for (unsigned char raw : std::string("1200")) key(raw);
                key(13);
                selectRow("PRESS E TO RUN SIM"); key('e');
                require(arbiter.getApplicationLayer() == TheArbiter::ApplicationLayer::ACTIVE_WORKSPACE, "Run Atomic");
                require(host.presentation().runtimeStatus.objectLine.find("1320") != std::string::npos, "1320 markers retain population semantics");
                tick();
                auto detail = [&](const std::string& token) {
                    for (const auto& line : host.presentation().runtimeStatus.detailLines)
                        if (line.find(token) != std::string::npos) return line;
                    throw std::runtime_error("Missing runtime detail: " + token);
                };
                auto menuCommand = [&](const std::string& token) {
                    for (const auto& item : host.menu().items)
                        if (item.enabled && item.label.find(token) != std::string::npos) return item.command;
                    throw std::runtime_error("Missing Atomic menu command: " + token);
                };
                auto selectVector = [&](const std::string& name) {
                    for (int i = 0; i < 5 && detail("FIELD VIEW:").find(name) == std::string::npos; ++i) key('v');
                    require(detail("FIELD VIEW:").find(name) != std::string::npos, "Vector field selection reaches requested view");
                };
                auto displayedMagnitude = [&]() {
                    const auto line = detail("FIELD VIEW:");
                    const auto maximum = line.find(" | MAX: ");
                    require(maximum != std::string::npos, "Selected field reports magnitude");
                    return std::stod(line.substr(maximum + 8));
                };
                require(host.handleMenuCommand(menuCommand("TEST FIRE")), "Atomic Layer-3 menu command handled");
                require(detail("TEST FIRE:").find("TEST FIRE: ON") != std::string::npos, "Fire mode actually toggles in Atomic");
                auto fire = [&]() {
                    require(host.handlePointerInput(arbiter.translateMouseButton(GLUT_LEFT_BUTTON, GLUT_DOWN, 640, 450)),
                        "Atomic Layer-3 pointer press handled");
                    require(host.handlePointerInput(arbiter.translateMouseButton(GLUT_LEFT_BUTTON, GLUT_UP, 640, 450)),
                        "Atomic Layer-3 pointer release handled");
                };
                key('1'); fire();
                require(detail("TEST FIRE:").find("ACTIVE DEBUG: 1/128") != std::string::npos,
                    "Center-ray click creates active electron in Atomic");
                selectVector("ELECTRIC_FIELD");
                require(detail("FIELD VIEW:").find("SCALE: LOG") != std::string::npos, "Electric field defaults to logarithmic magnitude");
                const double electricBefore = displayedMagnitude();
                require(electricBefore > 0.0, "Fired charge sources a nonzero sampled electric field");
                for (int i = 0; i < 30; ++i) tick();
                require(displayedMagnitude() > 0.0 && displayedMagnitude() != electricBefore,
                    "Electric field magnitudes refresh while projectile moves");
                key(' '); // Freeze source positions while comparing display backends.
                tick();
                const double frozenElectric = displayedMagnitude();
                require(detail("FIELD DISPLAY:").find("CUDA E/SCALAR") != std::string::npos,
                    "CUDA field display active after real drawing");
                key('h'); tick();
                require(detail("FIELD DISPLAY:").find("CPU REFERENCE") != std::string::npos &&
                    displayedMagnitude() == frozenElectric, "CPU display toggle preserves stored physics");
                require(host.handleMenuCommand(menuCommand("FIELD DISPLAY")), "Field display menu toggle handled");
                tick();
                require(detail("FIELD DISPLAY:").find("CUDA E/SCALAR") != std::string::npos &&
                    displayedMagnitude() == frozenElectric, "CUDA display restored without changing fields");
                key(' ');
                selectVector("MAGNETIC_FIELD"); tick();
                require(displayedMagnitude() > 0.0, "Moving electron sources magnetic diagnostics");
                key('2'); fire(); tick();
                key('3'); fire(); tick();
                require(detail("TEST FIRE:").find("ACTIVE DEBUG: 3/128") != std::string::npos,
                    "Electron, ion and neutral projectiles share active rendering path");
                for (const auto* name : {"ELECTRIC_FIELD", "MAGNETIC_FIELD", "CURRENT_DENSITY", "CURL_B"}) {
                    selectVector(name); tick();
                }
                for (unsigned char raw : std::string("vbbbggg123")) {
                    const auto routed = arbiter.routeKeyboard(keyboard.onKey(raw, 0, 0));
                    require(routed.workspaceInput.action == WorkspaceInputAction::RawKey, "Atomic field debug raw-key routing");
                    key(raw); tick();
                }
                require(host.handleMenuCommand(menuCommand("Clear test particles")), "Atomic clear command handled");
                require(detail("TEST FIRE:").find("ACTIVE DEBUG: 0/128") != std::string::npos, "Clear removes projectile sources");
                selectVector("ELECTRIC_FIELD");
                require(displayedMagnitude() == 0.0, "Clear refreshes electric field to zero");
                tick();
                require(cudaDeviceSynchronize() == cudaSuccess, "CUDA runtime synchronization");
                key('q'); tick();
                require(host.menu().items.empty() && !host.handlePointerInput(
                    arbiter.translateMouseButton(GLUT_LEFT_BUTTON, GLUT_DOWN, 640, 450)),
                    "Layer-3 test firing unavailable after returning to configuration");
                key('q'); tick();
                std::printf("PASS: Atomic size %d, independent field grid, 66 spawn choices, firing, dynamic E/B, CPU/CUDA field drawing\n",
                    renderer.getSimBoxSize());
            };
            runAtomic();
            selectRow("MULPHY_SIM SELECTION"); key('d'); tick();
            require(arbiter.getActiveWorkspace() == TheArbiter::WorkspaceId::MULTIPHYSICS_SIM &&
                host.presentation().statusTone == WorkspaceStatusTone::Warning,
                "Skeletal MULTIPHYSICS_SIM remains separately reachable and unavailable");
            key('a');
            selectRow("MULPHY_SIM SELECTION"); key('a'); tick();
            require(arbiter.getActiveWorkspace() == TheArbiter::WorkspaceId::PARTICLE_SIM, "ParticleSim reachable");
            selectRow("CONFIGURE WORKSPACE"); key('e');
            selectRow("PARTICLE AMOUNT"); key('d'); // nonempty population with unchanged default radius
            selectRow("PRESS E TO RUN SIM"); key('e'); tick();
            require(arbiter.getApplicationLayer() == TheArbiter::ApplicationLayer::ACTIVE_WORKSPACE, "ParticleSim runs at box 16");
            require(cudaDeviceSynchronize() == cudaSuccess, "ParticleSim CUDA update at box 16");
            key('q'); tick(); key('q'); tick(); key('q'); settle();
            require(arbiter.isGlobalShell(), "Q returns to shell");
            checkDomain(16);
            selectRow("DOMAIN SELECTION"); key('a');
            resize(32, 'e');
            unsupported();
            selectRow("DOMAIN SELECTION"); key('d');
            selectRow("CONFIG GLOBAL SHELL"); key('e'); settle();
            require(arbiter.isDomainSelection() && renderer.getSimBoxSize() == 32,
                "MULTIPHYSICS domain entry preserves committed 32/128");
            checkDomain(32);
            runAtomic();
            checkDomain(32);
            std::puts("PASS: size-32 Atomic runtime, field debug keys and all seven field geometries");
            selectRow("MULPHY_SIM SELECTION"); key('a'); tick();
            require(arbiter.getActiveWorkspace() == TheArbiter::WorkspaceId::PARTICLE_SIM, "ParticleSim reachable at size 32");
            selectRow("CONFIGURE WORKSPACE"); key('e');
            selectRow("PARTICLE AMOUNT"); key('d');
            selectRow("PRESS E TO RUN SIM"); key('e'); tick();
            require(arbiter.getApplicationLayer() == TheArbiter::ApplicationLayer::ACTIVE_WORKSPACE, "Nonempty ParticleSim runs at box 32");
            require(cudaDeviceSynchronize() == cudaSuccess, "ParticleSim CUDA update at box 32");
            checkDomain(32);
            std::puts("PASS: size-32 ParticleSim runtime and CUDA synchronization");
            key('q'); tick(); key('q'); tick(); key('q'); settle();
            require(arbiter.isGlobalShell(), "Size-32 runtime returns to shell");
            selectRow("DOMAIN SELECTION"); key('a');
            resize(16, 13);
            resize(8, 13);
            selectRow("DOMAIN SELECTION"); key('d');
            selectRow("CONFIG GLOBAL SHELL"); key('e'); settle();
            runAtomic();
            key('q'); settle();
            require(arbiter.isGlobalShell(), "Size-8 Atomic runtime returns to shell");
            selectRow("DOMAIN SELECTION"); key('a');
            resize(4, 13);
            resize(16, 'e'); // 4 -> 8 -> 16 under one input lock
            resize(4, 13); // 16 -> 8 -> 4
            resize(32, 'e'); // 4 -> 8 -> 16 -> 32 under one input lock
            resize(4, 13); // 32 -> 16 -> 8 -> 4
            unsupported();
            selectRow("DOMAIN SELECTION"); key('d');
            selectRow("CONFIG GLOBAL SHELL"); key(13); settle();
            runAtomic();
            key('q'); settle();
            checkDomain(4);
            std::puts("PASS: resize sequences, center crossing, input lock, geometry propagation, camera scaling, 4/8/16/32 chained presets and runtime");
            host.shutdown();
            std::puts("PASS: Layer 2/3, field debug routing/rendering, CUDA sync, repeated return, ESC routing");
        }
        glutDestroyWindow(window);
        std::puts("PASS: normal integration-process cleanup");
        return 0;
    }
    catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
