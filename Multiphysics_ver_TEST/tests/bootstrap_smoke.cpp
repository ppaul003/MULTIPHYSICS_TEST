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
                glTranslatef(0.0f, 0.0f, -8.0f);
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
            tick();
            require(arbiter.isGlobalShell() && value(0) == "IDLE", "Startup IDLE");
            require(host.presentation().sections.at(0).rows.size() == 3, "Exactly three shell rows");
            require(
                host.presentation().statusLine == "SIM SIZE READY 4: (64^3)",
                "Default simulation size ready"
            );

            require(
                host.presentation().statusTone == WorkspaceStatusTone::Ready,
                "Default simulation size reports ready"
            );
            key('w'); key('e');
            require(arbiter.isGlobalShell() && !host.domainTransitionActive(), "IDLE activation blocked");
            key('w');
            for (const char* preset : {"8", "16", "32", "2", "4"}) {
                key('d'); require(value(1) == preset, "Preset cycle");
            }
            require(host.presentation().capabilityLines.at(1).tone == WorkspaceStatusTone::Transition,
                "Selected supported preset is orange");
            key('d'); // Leave an unsupported IDLE preset selected to check hidden-state independence.
            key('w'); key('d');
            require(value(0) == "MULTIPHYSICS" && value(1) == "METRIC", "MULTIPHYSICS metric shell");
            key('d'); require(value(0) == "IDLE", "Domain cycle wraps");
            key('a'); require(value(0) == "MULTIPHYSICS", "Reverse domain cycle");
            key('s'); key('d');
            require(value(1) == "IMPERIAL" && host.presentation().statusTone == WorkspaceStatusTone::Warning,
                "Imperial warning");
            key('s'); key('e'); key(13);
            require(arbiter.isGlobalShell() && !host.domainTransitionActive(), "Imperial blocks E and Enter");
            key('w'); key('a');
            require(host.presentation().statusTone == WorkspaceStatusTone::Ready, "Metric ready");
            key('s'); key('e');
            require(host.domainTransitionActive(), "Metric E starts transition despite IDLE preset 8");
            settle();
            require(arbiter.isDomainSelection() && arbiter.getActiveWorkspace() == TheArbiter::WorkspaceId::MULTIPHYSICS_SIM,
                "Enter MULTIPHYSICS Layer 1");
            std::puts("PASS: shell rows, presets, domain cycle, metric/imperial guards, entry and rendering");

            // Existing production workspace configuration and debug keys.
            key('w'); key('e');
            require(arbiter.getApplicationLayer() == TheArbiter::ApplicationLayer::WORKSPACE_CONFIGURATION,
                "Configure MULTIPHYSICS Layer 2");
            key('s'); key('e');
            for (unsigned char raw : std::string("1200")) key(raw);
            key(13);
            key('w'); key('w'); key('e');
            require(arbiter.getApplicationLayer() == TheArbiter::ApplicationLayer::ACTIVE_WORKSPACE,
                "Start existing Layer 3 runtime");
            require(host.presentation().runtimeStatus.objectLine.find("1320") != std::string::npos,
                "1200 Argon at 10 percent ionization creates 1320 total markers");
            tick();
            for (unsigned char raw : std::string("fvbcg123")) {
                const auto routed = arbiter.routeKeyboard(keyboard.onKey(raw, 0, 0));
                require(routed.workspaceInput.action == WorkspaceInputAction::RawKey, "Field debug raw key parity");
                key(raw); tick();
            }
            require(cudaDeviceSynchronize() == cudaSuccess, "CUDA runtime synchronization");
            key('q'); tick(); key('q'); tick(); key('q'); settle();
            require(arbiter.isGlobalShell(), "Q returns through Layer 2 and Layer 1 to shell");
            key(13); settle();
            require(arbiter.isDomainSelection(), "Enter key re-enters workspace");
            key('s'); key('a'); tick();
            require(arbiter.getActiveWorkspace() == TheArbiter::WorkspaceId::PARTICLE_SIMULATION,
                "Existing ParticleSim cartridge remains reachable");
            key('q'); settle();
            require(arbiter.isGlobalShell(), "Repeated return to shell");
            require(arbiter.routeKeyboard(keyboard.onKey(27, 0, 0)).arbiterCommand == TheArbiter::ArbiterCommand::CMD_EXIT,
                "ESC maps to exit command");
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
