#ifdef _WIN32
#include <direct.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <cctype>
#include <sstream>
#include <iomanip>

#include "kernel.h"
#include "EuclidEngineEM.h"

#include <algorithm>
#include <cstdio>

#ifdef _WIN32

namespace {

    HICON g_vitruGenLargeIcon = nullptr;
    HICON g_vitruGenSmallIcon = nullptr;

    bool applyVitruGenIconFromFile(
        const char* iconFilename) {

        if (!iconFilename || iconFilename[0] == '\0') {

            return false;
        }

        // Immediately after glutCreateWindow(), this should
        // resolve to the FreeGLUT window.
        HWND windowHandle = GetActiveWindow();

        // Fallback lookup using the original window title.
        if (!windowHandle) {

            windowHandle =
                FindWindowA(nullptr, "ANAHEIM SYSTEMS DYNAMICS | Simulation Module Test & Debug | Development Ver. TEST");
        }

        if (!windowHandle) {

            printf(
                "[EuclidEngine] WARNING: "
                "Could not locate the VitruGen window.\n"
            );

            return false;
        }

        const int largeWidth = GetSystemMetrics(SM_CXICON);
        const int largeHeight = GetSystemMetrics(SM_CYICON);
        const int smallWidth = GetSystemMetrics(SM_CXSMICON);
        const int smallHeight = GetSystemMetrics(SM_CYSMICON);

        g_vitruGenLargeIcon =
            static_cast<HICON>(
                LoadImageA(
                    nullptr,
                    iconFilename,
                    IMAGE_ICON,
                    largeWidth,
                    largeHeight,
                    LR_LOADFROMFILE)
                );

        g_vitruGenSmallIcon =
            static_cast<HICON>(
                LoadImageA(
                    nullptr,
                    iconFilename,
                    IMAGE_ICON,
                    smallWidth,
                    smallHeight,
                    LR_LOADFROMFILE
                )
                );

        if (!g_vitruGenLargeIcon &&
            !g_vitruGenSmallIcon) {

            printf(
                "[EuclidEngine] WARNING: "
                "Could not load app icon: %s\n",
                iconFilename
            );

            return false;
        }

        if (g_vitruGenLargeIcon) {

            SendMessageA(
                windowHandle,
                WM_SETICON,
                ICON_BIG,
                reinterpret_cast<LPARAM>(g_vitruGenLargeIcon)
            );
        }

        if (g_vitruGenSmallIcon) {

            SendMessageA(
                windowHandle,
                WM_SETICON,
                ICON_SMALL,
                reinterpret_cast<LPARAM>(g_vitruGenSmallIcon)
            );
        }

        printf(
            "[EuclidEngine] Application icon loaded: %s\n",
            iconFilename
        );

        return true;
    }

    void releaseVitruGenWindowIcons() {

        if (g_vitruGenLargeIcon) {

            DestroyIcon(g_vitruGenLargeIcon);
            g_vitruGenLargeIcon = nullptr;
        }

        if (g_vitruGenSmallIcon) {

            DestroyIcon(g_vitruGenSmallIcon);
            g_vitruGenSmallIcon = nullptr;
        }
    }

} // namespace

#endif

EuclidEngine* EuclidEngine::s_instance = nullptr;

EuclidEngine::~EuclidEngine() {
    shutdown();
}

bool EuclidEngine::init(int argc, char** argv) {
    std::printf("ANAHEIM SYSTEMS DYNAMICS\n");
    std::printf("Simulation Module Test & Debug\n");
    std::printf("Development Ver. TEST\n\n");

    s_instance = this;

    initGL(&argc, argv);
    initRenderer();

    if (!initWorkspaceHost()) {
        std::printf("[EuclidEngine] ERROR: workspace host initialization failed.\n");
        return false;
    }

    glutDisplayFunc(&EuclidEngine::sDisplay);
    glutReshapeFunc(&EuclidEngine::sReshape);
    glutKeyboardFunc(&EuclidEngine::sKeyboard);
    glutKeyboardUpFunc(&EuclidEngine::sKeyboardUp);
    glutMouseFunc(&EuclidEngine::sMouse);
    glutMotionFunc(&EuclidEngine::sMotion);
    glutIdleFunc(&EuclidEngine::sIdle);
    glutCloseFunc(&EuclidEngine::sClose);
    glutEntryFunc(&EuclidEngine::sEntry);
    glutMenuStatusFunc(&EuclidEngine::sMenuStatus);
    initMenus();

    return true;
}

void EuclidEngine::initGL(int* argc, char** argv) {
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(kWidth, kHeight);
    glutCreateWindow("ANAHEIM SYSTEMS DYNAMICS | Simulation Module Test & Debug | Development Ver. TEST");
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);

#ifdef _WIN32

    applyVitruGenIconFromFile("anaheim.ico");
    m_windowHandle = GetActiveWindow();

#endif

    glewExperimental = GL_TRUE;
    const GLenum glewResult = glewInit();
    if (glewResult != GLEW_OK) {
        std::printf("[EuclidEngine] WARNING: GLEW initialization failed: %s\n",
            glewGetErrorString(glewResult));
    }

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.05f, 0.05f, 0.15f, 1.0f);

    m_viewport.resize(kWidth, kHeight);
    m_viewport.applyPerspective(60.0f);
}

void EuclidEngine::initRenderer() {
    if (m_renderer) return;

    m_renderer = new EuclidRenderer();
    m_renderer->setWindowSize(m_viewport.getWidth(), m_viewport.getHeight());
    m_renderer->setFOV(60.0f);
    m_renderer->setSimBoxSize(4);

    std::printf("[EuclidEngine] Grid-only EuclidRenderer initialized.\n");
}

bool EuclidEngine::initWorkspaceHost() {
    if (!m_renderer) return false;

    WorkspaceServices services;
    services.renderer = m_renderer;
    services.arbiter = &m_arbiter;
    services.viewport = &m_viewport;
    services.camera = &m_camera;

    m_arbiter.setApplicationLayer(TheArbiter::ApplicationLayer::GLOBAL_SHELL);
    m_arbiter.setWorkspaceDomain(TheArbiter::WorkspaceDomain::NONE);
    m_arbiter.setActiveWorkspace(TheArbiter::WorkspaceId::DIAGNOSTIC);
    m_camera.setBehaviorMode(CameraProcessor::CAM_MENU_PREVIEW);

    return m_tesseract.initialize(services);
}

void EuclidEngine::initMenus() {
    glutDetachMenu(GLUT_RIGHT_BUTTON);
    if (m_menuId != 0) {
        glutDestroyMenu(m_menuId);
        m_menuId = 0;
    }
    m_menuId = glutCreateMenu(&EuclidEngine::sMainMenu);
    m_workspaceMenuCommands.clear();
    glutAddMenuEntry("=========================================", MENU_NOP);
    glutAddMenuEntry("- Simulation Module Test & Debug -", MENU_NOP);
    glutAddMenuEntry("=========================================", MENU_NOP);
    const WorkspaceMenuPresentation workspaceMenu = m_tesseract.menu();
    for (const WorkspaceMenuItem& item : workspaceMenu.items) {
        const int slot = static_cast<int>(m_workspaceMenuCommands.size());
        m_workspaceMenuCommands.push_back(item.command);
        glutAddMenuEntry(item.label.c_str(),
            item.enabled ? MENU_WORKSPACE_COMMAND_BASE + slot : MENU_NOP);
    }
    glutAddMenuEntry("* Quit (esc)", MENU_QUIT);
    glutAttachMenu(GLUT_RIGHT_BUTTON);
    m_menuDirty = false;
}

void EuclidEngine::rebuildMenus() {
    // Never invalidate the slot mapping while a native menu is being used.
    if (m_menuOpen) {
        m_menuDirty = true;
        return;
    }
    initMenus();
}

WorkspaceFrameContext
EuclidEngine::buildWorkspaceFrameContext(float deltaTime) const {
    WorkspaceFrameContext frame;
    frame.deltaTime = deltaTime;
    frame.elapsedTime = static_cast<float>(glutGet(GLUT_ELAPSED_TIME)) * 0.001f;
    frame.viewportWidth = m_viewport.getWidth();
    frame.viewportHeight = m_viewport.getHeight();
    frame.displayEnabled = m_displayEnabled;
    return frame;
}

void EuclidEngine::run() {
    if (!m_exiting) glutMainLoop();
}

void EuclidEngine::shutdown() {
    if (m_cleaned) return;
    m_cleaned = true;

    cancelInput();
    if (m_menuId != 0) {
        glutDetachMenu(GLUT_RIGHT_BUTTON);
        glutDestroyMenu(m_menuId);
        m_menuId = 0;
    }
    m_tesseract.shutdown();

    delete m_renderer;
    m_renderer = nullptr;
    s_instance = nullptr;
#ifdef _WIN32
    releaseVitruGenWindowIcons();
#endif
}

void EuclidEngine::requestExit() {
    if (m_exiting) return;
    m_exiting = true;
    glutLeaveMainLoop();
}

void EuclidEngine::computeFPS() {
    static int frameCount = 0;
    static int previousTime = glutGet(GLUT_ELAPSED_TIME);

    ++frameCount;
    const int currentTime = glutGet(GLUT_ELAPSED_TIME);
    const int elapsed = currentTime - previousTime;
    if (elapsed < 1000) return;

    const float fps = static_cast<float>(frameCount) * 1000.0f /
        static_cast<float>(elapsed);

    char title[128];
    std::snprintf(title, sizeof(title), "ANAHEIM SYSTEMS DYNAMICS | Simulation Module Test & Debug | Development Ver. TEST : %.1f fps", fps);
    glutSetWindowTitle(title);

    frameCount = 0;
    previousTime = currentTime;
}

void EuclidEngine::sDisplay() {
    if (s_instance) s_instance->onDisplay();
}

void EuclidEngine::sReshape(int w, int h) {
    if (s_instance) s_instance->onReshape(w, h);
}

void EuclidEngine::sKeyboard(unsigned char key, int x, int y) {
    if (s_instance) s_instance->onKeyboard(key, x, y);
}

void EuclidEngine::sKeyboardUp(unsigned char key, int x, int y) {
    if (s_instance) s_instance->onKeyboardUp(key, x, y);
}

void EuclidEngine::sMainMenu(int value) {
    if (!s_instance || value == MENU_NOP) return;
    if (value >= MENU_WORKSPACE_COMMAND_BASE) {
        const int slot = value - MENU_WORKSPACE_COMMAND_BASE;
        if (slot >= 0 && slot < static_cast<int>(s_instance->m_workspaceMenuCommands.size())) {
            s_instance->m_tesseract.handleMenuCommand(s_instance->m_workspaceMenuCommands[slot]);
            s_instance->rebuildMenus();
            glutPostRedisplay();
        }
        return;
    }
    if (value == MENU_QUIT) s_instance->requestExit();
}

void EuclidEngine::sMenuStatus(int status, int, int) {
    if (!s_instance) return;
    s_instance->m_menuOpen = status == GLUT_MENU_IN_USE;
    if (s_instance->m_menuOpen) s_instance->cancelInput();
    // A deferred rebuild runs from idle AFTER the selected command callback.
}

void EuclidEngine::sEntry(int state) {
    if (s_instance && state == GLUT_LEFT) s_instance->cancelInput();
}

void EuclidEngine::sMouse(int button, int state, int x, int y) {

    if (s_instance)
        s_instance->onMouse(button, state, x, y);
}

void EuclidEngine::sMotion(int x, int y) {

    if (s_instance)
        s_instance->onMotion(x, y);
}

void EuclidEngine::sIdle() {
    if (s_instance) s_instance->onIdle();
}

void EuclidEngine::sClose() {
    if (s_instance) s_instance->onClose();
}

void EuclidEngine::onReshape(int w, int h) {
    cancelInput();
    m_viewport.resize(w, h);
    m_viewport.applyPerspective(60.0f);

    if (m_renderer)
        m_renderer->setWindowSize(w, h);

    glutPostRedisplay();
}

void EuclidEngine::onDisplay() {
    glEnable(GL_DEPTH_TEST);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    m_viewport.applyPerspective(60.0f);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    m_camera.updateLag();
    m_camera.applyCameraTransform();

    const WorkspaceFrameContext frame = buildWorkspaceFrameContext(0.0f);
    m_tesseract.render(frame);
    m_viewport.drawOverlay(m_tesseract.presentation());
    m_tesseract.renderOverlay(frame);

    glutSwapBuffers();
}

void EuclidEngine::onKeyboard(unsigned char key, int x, int y) {
    const KeyboardInput::KeyEvent event = m_keyboard.onKey(key, x, y);
    TheArbiter::ArbiterResult result = m_arbiter.routeKeyboard(event);
    const unsigned char physicalKey = static_cast<unsigned char>(std::tolower(key));
    result.workspaceInput.repeated = m_keysDown[physicalKey];
    m_keysDown[physicalKey] = true;

    if (result.arbiterCommand == TheArbiter::ArbiterCommand::CMD_EXIT) {
        requestExit();
        return;
    }

    if (result.hasWorkspaceInput) {
        if (m_tesseract.handleInput(result.workspaceInput)) rebuildMenus();
        glutPostRedisplay();
    }
}

void EuclidEngine::onKeyboardUp(unsigned char key, int x, int y) {
    m_keysDown[static_cast<unsigned char>(std::tolower(key))] = false;
    const auto result = m_arbiter.routeKeyboard(m_keyboard.onKey(key, x, y));
    if (result.hasWorkspaceInput) m_tesseract.handleInputRelease(result.workspaceInput);
}

void EuclidEngine::cancelInput() {
    m_tesseract.cancelInput();
    m_mouse.onButton(GLUT_LEFT_BUTTON, GLUT_UP, 0, 0);
    m_workspacePointerCaptured = false;
}

void EuclidEngine::onMouse(int button, int state, int x, int y) {
    if (button == GLUT_RIGHT_BUTTON || m_menuOpen) return;

    m_mouse.onButton(button, state, x, y);

    // Never manipulate camera during automatic transition.
    if (m_tesseract.domainTransitionActive())
        return;

    const bool wasCaptured = m_workspacePointerCaptured;
    const bool handled = m_tesseract.handlePointerInput(
        m_arbiter.translateMouseButton(button, state, x, y));
    if (button == GLUT_LEFT_BUTTON)
        m_workspacePointerCaptured = state == GLUT_DOWN && handled;
    if (handled || (button == GLUT_LEFT_BUTTON && wasCaptured)) {
        glutPostRedisplay();
        return;
    }

    // Unconsumed workspace input retains the existing generic orbit/zoom path.
    if (!m_arbiter.isWorkspaceLayer())
        return;

    // FreeGLUT wheel up/down.
    if (state == GLUT_DOWN) {

        if (button == 3 && m_camera.zoomEnabled()) {
            m_camera.zoom(+0.05f);
        }

        else if (button == 4 && m_camera.zoomEnabled()) {
            m_camera.zoom(-0.05f);
        }
    }

    glutPostRedisplay();
}

void EuclidEngine::onMotion(int x, int y) {

    int dx = 0;
    int dy = 0;

    if (!m_mouse.onMotion(x, y, dx, dy))
        return;

    if (m_tesseract.domainTransitionActive())
        return;

    if (!m_arbiter.isWorkspaceLayer())
        return;

    if (m_menuOpen) return;
    if (m_tesseract.handlePointerInput(m_arbiter.translateMouseMotion(x, y, dx, dy)) ||
        m_workspacePointerCaptured) {
        glutPostRedisplay();
        return;
    }

    if (!m_camera.orbitEnabled())
        return;

    m_camera.orbit(
        static_cast<float>(dx),
        static_cast<float>(dy)
    );

    glutPostRedisplay();
}

void EuclidEngine::onIdle() {
    static int previousTimeMs = glutGet(GLUT_ELAPSED_TIME);

    const int currentTimeMs = glutGet(GLUT_ELAPSED_TIME);
    const int elapsedMs = currentTimeMs - previousTimeMs;
    previousTimeMs = currentTimeMs;

    float deltaTime = static_cast<float>(elapsedMs) * 0.001f;
    deltaTime = (std::min)(deltaTime, 0.050f);

    const WorkspaceFrameContext frame = buildWorkspaceFrameContext(deltaTime);
#ifdef _WIN32
    // FreeGLUT has no keyboard-focus-loss callback. Stop transient input when
    // this host window loses focus, including Alt-Tab without pointer movement.
    if (m_windowHandle && GetForegroundWindow() != m_windowHandle && !m_menuOpen) {
        cancelInput();
        std::fill(std::begin(m_keysDown), std::end(m_keysDown), false);
    }
#endif
    const auto previousLayer = m_arbiter.getApplicationLayer();
    const auto previousWorkspace = m_arbiter.getActiveWorkspace();
    const bool wasTransitioning = m_tesseract.domainTransitionActive();
    m_tesseract.update(frame);
    if (previousLayer != m_arbiter.getApplicationLayer() ||
        previousWorkspace != m_arbiter.getActiveWorkspace() ||
        wasTransitioning != m_tesseract.domainTransitionActive())
        rebuildMenus();
    if (m_menuDirty && !m_menuOpen) rebuildMenus();

    computeFPS();
    glutPostRedisplay();
}

void EuclidEngine::onClose() {
    shutdown();
}
