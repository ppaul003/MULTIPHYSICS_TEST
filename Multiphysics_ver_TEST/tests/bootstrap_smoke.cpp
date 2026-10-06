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
#include <set>
#include <cmath>

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

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
            // Destruction order keeps the GL context and renderer alive for CUDA cleanup.
            EuclidRenderer renderer;
            renderer.setWindowSize(1280, 900);
            renderer.setFOV(60.0f);
            renderer.setSimBoxSize(4);
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
            const float particleRadius = host.particleDomainState().collisionRadius;
            const float plasmaRadius = host.multiphysicsDomainState().collisionRadius;
            auto checkDomain = [&](int box) {
                for (const auto& state : {host.particleDomainState(), host.multiphysicsDomainState()}) {
                    require(state.collisionDimensions == glm::ivec3(64), "CUDA remains 64^3");
                    require(state.collisionOrigin == glm::vec3(-box * 0.5f), "CUDA origin follows committed box");
                    require(state.collisionCellSize == glm::vec3(box / 64.0f), "CUDA collision spacing follows committed box");
                    require(state.physicalGrid.dimensions == glm::ivec3(8), "Physical grid remains 8^3");
                    require(state.physicalGrid.origin == glm::vec3(-box * 0.5f), "Physical origin follows committed box");
                    require(state.physicalGrid.voxelEdgeM == box / 8.0f, "Physical voxel spacing follows committed box");
                    SpawnDensityRegionGrid3D spawnGrid;
                    SpawnDensityRegion3D region;
                    require(spawnGrid.centeredRegion(state.physicalGrid, region), "Center spawn region remains valid");
                    require(region.volumeM3 == (box == 8 ? 8.0f : 1.0f), "2x2x2 spawn volume follows physical geometry");
                }
                const auto fields = host.multiphysicsDomainState();
                require(fields.fieldGeometryValid && fields.fieldCellCount == 512, "All seven borrowed field geometries valid with 512 cells");
                require(host.particleDomainState().collisionRadius == particleRadius && fields.collisionRadius == plasmaRadius,
                    "Domain resizing leaves collision radii unchanged");
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
                selectRow("SIMULATION BOX SIZE");
                for (int i = 0; i < 5 && value(1) != std::to_string(target); ++i) key('d');
                require(value(1) == std::to_string(target), "Requested size displayed");
                require(host.presentation().statusLine == "SIM SIZE SELECT: " + std::to_string(target) + " (64^3)" &&
                    host.presentation().statusTone == WorkspaceStatusTone::Transition, "Pending size orange");
                require(renderer.getSimBoxSize() == from, "Selection does not commit");
                checkDomain(from);
                glm::vec3 eyeBefore, directionBefore;
                camera.getCenterViewRay(eyeBefore, directionBefore);
                selectRow("CONFIG GLOBAL SHELL"); key(commitKey);
                require(host.boxResizeActive() && !host.domainTransitionActive(), "Dedicated resize transition begins");
                require(host.boxResizeState().phase == Phase::SetHold, "Brief hold starts resize");
                std::set<Phase> visited;
                std::vector<Phase> sequence;
                bool boundaryIntermediate = false, planeIntermediate = false, gridIntermediate = false;
                float frozenSlice = -1.0f;
                auto previous = host.boxResizeState();
                for (int i = 0; i < 2400 && host.boxResizeActive(); ++i) {
                    const auto state = host.boxResizeState();
                    visited.insert(state.phase);
                    if (sequence.empty() || sequence.back() != state.phase) sequence.push_back(state.phase);
                    const auto presentation = host.presentation();
                    require(presentation.statusBlink && presentation.frameBlink &&
                        presentation.frameTone == WorkspaceStatusTone::Transition, "Resize status/frame blink orange");
                    require(renderer.getSimBoxSize() == from, "Renderer commits only at completion");
                    if (i % 30 == 0) {
                        for (unsigned char raw : std::string("wsadeq\r")) key(raw);
                        require(value(1) == std::to_string(target) && value(0) == "IDLE" &&
                            host.presentation().sections.at(0).rows.at(2).selected, "Resize ignores menu input");
                        require(arbiter.routeKeyboard(keyboard.onKey(27, 0, 0)).arbiterCommand == TheArbiter::ArbiterCommand::CMD_EXIT,
                            "ESC still exits through Arbiter while resize is active");
                        checkDomain(from);
                    }
                    if (state.slicePaused) {
                        if (frozenSlice < 0) frozenSlice = state.sliceTravel;
                        require(std::fabs(std::fmod(state.sliceTravel, 3.0f) - 0.5f) < 1e-5f &&
                            state.sliceTravel == frozenSlice, "Slice freezes exactly at XY center");
                    }
                    if (state.phase == Phase::WaitForExpandSliceCenter || state.phase == Phase::WaitForShrinkSliceCenter)
                        require(!state.slicePaused, "Slice keeps moving while waiting for next crossing");
                    const auto between = [&](float extent) { return extent > 4.0f && extent < 8.0f; };
                    boundaryIntermediate |= between(state.boundarySize);
                    planeIntermediate |= between(state.planeSize);
                    gridIntermediate |= between(state.gridSize);
                    if (state.phase == Phase::ExpandBoundary)
                        require(state.planeSize == 4 && state.gridSize == 4, "Boundary expands around old plane/grid");
                    if (state.phase == Phase::ExpandPlane)
                        require(state.boundarySize == 8 && state.gridSize == 4, "Plane expands before grid");
                    if (state.phase == Phase::ExpandGrid)
                        require(state.boundarySize == 8 && state.planeSize == 8, "Grid expands last");
                    if (state.phase == Phase::ShrinkGrid)
                        require(state.boundarySize == 8 && state.planeSize == 8, "Grid shrinks first");
                    if (state.phase == Phase::ShrinkPlane)
                        require(state.boundarySize == 8 && state.gridSize == 4, "Plane shrinks second");
                    if (state.phase == Phase::ShrinkBoundary)
                        require(state.planeSize == 4 && state.gridSize == 4, "Boundary shrinks last");
                    // No discontinuities in any visual extent between animation ticks.
                    require(std::fabs(state.boundarySize - previous.boundarySize) < 0.2f &&
                        std::fabs(state.planeSize - previous.planeSize) < 0.2f &&
                        std::fabs(state.gridSize - previous.gridSize) < 0.2f, "Smooth visual extents");
                    previous = state;
                    // Render every tick: actual legacy GL paths with independent sizes.
                    tick();
                }
                require(!host.boxResizeActive(), "Resize completes within bounded time");
                const std::vector<Phase> expected = target == 8
                    ? std::vector<Phase>{Phase::SetHold, Phase::ExpandBoundary, Phase::WaitForExpandSliceCenter, Phase::ExpandPlane, Phase::ExpandGrid}
                    : std::vector<Phase>{Phase::SetHold, Phase::WaitForShrinkSliceCenter, Phase::ShrinkGrid, Phase::ShrinkPlane, Phase::ShrinkBoundary};
                require(sequence == expected, "Expansion/shrink phases occur in required order");
                require(boundaryIntermediate && planeIntermediate && gridIntermediate, "All extents interpolate");
                require(renderer.getSimBoxSize() == target && host.boxResizeState().activeSize == target, "Final domain committed");
                require(host.presentation().statusLine == "SIM SIZE READY: " + std::to_string(target) + " (64^3)" &&
                    host.presentation().statusTone == WorkspaceStatusTone::Ready && !host.presentation().statusBlink &&
                    !host.presentation().frameBlink, "Final ready state without blink");
                checkDomain(target);
                glm::vec3 eyeAfter, directionAfter;
                camera.getCenterViewRay(eyeAfter, directionAfter);
                require(glm::length(directionBefore - directionAfter) < 1e-5f &&
                    glm::length(eyeAfter - eyeBefore * (float(target) / from)) < 1e-4f, "Camera scales all axes, preserves angle");
                const float center = host.boxResizeState().sliceTravel;
                tick();
                require(host.boxResizeState().sliceTravel > center && !host.boxResizeState().slicePaused, "Slice resumes toward +Z");
                key(commitKey); require(!host.boxResizeActive(), "Committing active size remains ready");
            };
            // Start after the first XY midpoint: the wait must seek a future crossing.
            for (int i = 0; i < 130; ++i) host.update(frame);
            resize(8, 'e');
            require(host.boxResizeState().sliceTravel > 3.5f, "Wait skips already-passed XY center");

            auto unsupported = [&]() {
                const int active = renderer.getSimBoxSize();
                for (int unavailable : {2, 16, 32}) {
                    selectRow("SIMULATION BOX SIZE");
                    for (int i = 0; i < 5 && value(1) != std::to_string(unavailable); ++i) key('d');
                    require(host.presentation().statusTone == WorkspaceStatusTone::Warning, "Unsupported size warning");
                    selectRow("CONFIG GLOBAL SHELL"); key('e'); key(13);
                    require(!host.boxResizeActive() && !host.domainTransitionActive() && renderer.getSimBoxSize() == active,
                        "Unsupported presets never mutate committed geometry");
                    checkDomain(active);
                }
            };
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
            checkDomain(8);
            require(renderer.getSimBoxSize() == 8, "Domain entry never forces size 4");

            auto runPlasma = [&]() {
                selectRow("CONFIGURE WORKSPACE"); key('e');
                require(arbiter.getApplicationLayer() == TheArbiter::ApplicationLayer::WORKSPACE_CONFIGURATION, "Configure plasma");
                selectRow("TOTAL GAS DENSITY"); key('e');
                for (unsigned char raw : std::string("1200")) key(raw);
                key(13);
                selectRow("PRESS E TO RUN SIM"); key('e');
                require(arbiter.getApplicationLayer() == TheArbiter::ApplicationLayer::ACTIVE_WORKSPACE, "Run plasma");
                require(host.presentation().runtimeStatus.objectLine.find("1320") != std::string::npos, "1320 markers retain population semantics");
                tick();
                for (unsigned char raw : std::string("fvbcg123")) {
                    const auto routed = arbiter.routeKeyboard(keyboard.onKey(raw, 0, 0));
                    require(routed.workspaceInput.action == WorkspaceInputAction::RawKey, "Field debug routing");
                    key(raw); tick();
                }
                require(cudaDeviceSynchronize() == cudaSuccess, "CUDA runtime synchronization");
                key('q'); tick(); key('q'); tick();
            };
            runPlasma();
            selectRow("MULPHY_SIM SELECTION"); key('a'); tick();
            require(arbiter.getActiveWorkspace() == TheArbiter::WorkspaceId::PARTICLE_SIMULATION, "ParticleSim reachable");
            selectRow("CONFIGURE WORKSPACE"); key('e');
            selectRow("PARTICLE AMOUNT"); key('d'); // nonempty population with unchanged default radius
            selectRow("PRESS E TO RUN SIM"); key('e'); tick();
            require(arbiter.getApplicationLayer() == TheArbiter::ApplicationLayer::ACTIVE_WORKSPACE, "ParticleSim runs at box 8");
            require(cudaDeviceSynchronize() == cudaSuccess, "ParticleSim CUDA update at box 8");
            key('q'); tick(); key('q'); tick(); key('q'); settle();
            require(arbiter.isGlobalShell(), "Q returns to shell");
            checkDomain(8);
            selectRow("DOMAIN SELECTION"); key('a');
            resize(4, 13);
            unsupported();
            selectRow("DOMAIN SELECTION"); key('d');
            selectRow("CONFIG GLOBAL SHELL"); key(13); settle();
            runPlasma();
            key('q'); settle();
            checkDomain(4);
            std::puts("PASS: resize sequences, center crossing, input lock, geometry propagation, camera scaling, box 8/4 runtime");
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
