#include "TheArbiterEM.h"

TheArbiter::TheArbiter() = default;

TheArbiter::ArbiterResult
TheArbiter::routeKeyboard(const KeyboardInput::KeyEvent& event) const {
    ArbiterResult result;
    result.workspaceInput.rawKey = event.rawKey;
    result.workspaceInput.x = event.x;
    result.workspaceInput.y = event.y;

    if (event.signal == KeyboardInput::KEY_ESCAPE) {
        result.handled = true;
        result.arbiterCommand = ArbiterCommand::CMD_EXIT;
        return result;
    }

    switch (event.signal) {
    case KeyboardInput::KEY_W:
        result.workspaceInput.action = WorkspaceInputAction::Previous;
        break;
    case KeyboardInput::KEY_S:
        result.workspaceInput.action = WorkspaceInputAction::Next;
        break;
    case KeyboardInput::KEY_A:
        result.workspaceInput.action = WorkspaceInputAction::Decrease;
        break;
    case KeyboardInput::KEY_D:
        result.workspaceInput.action = WorkspaceInputAction::Increase;
        break;
    case KeyboardInput::KEY_E:
    case KeyboardInput::KEY_ENTER:
        result.workspaceInput.action = WorkspaceInputAction::Activate;
        break;
    case KeyboardInput::KEY_Q:
        result.workspaceInput.action = WorkspaceInputAction::Back;
        break;
    case KeyboardInput::KEY_TAB:
        result.workspaceInput.action = WorkspaceInputAction::TogglePanel;
        break;
    case KeyboardInput::KEY_SPACE:
        result.workspaceInput.action = WorkspaceInputAction::TogglePause;
        break;
    default: {
        const unsigned char key = static_cast<unsigned char>(std::tolower(event.rawKey));

        const bool multiphysicsDebugKey = m_navigation.layer ==
            ApplicationLayer::ACTIVE_WORKSPACE &&
            m_navigation.workspace == WorkspaceId::ATOMIC_PARTICLES &&
            (key == 'f' || key == 'v' || key == 'b' || key == 'c' || key == 'g' || key == 'h');

        if ((event.rawKey >= '0' && event.rawKey <= '9') ||

            event.rawKey == 8 || event.rawKey == 127 || multiphysicsDebugKey) {
            result.workspaceInput.action = WorkspaceInputAction::RawKey;
        }
        else {

            result.workspaceInput.action = WorkspaceInputAction::None;
        }
        break;
    }
    }

    if (result.workspaceInput.action != WorkspaceInputAction::None) {
        result.hasWorkspaceInput = true;
        result.handled = true;
    }

    return result;
}

WorkspacePointerEvent TheArbiter::translateMouseButton(
    int button, int state, int x, int y) const {
    WorkspacePointerEvent input;
    input.pressed = state == GLUT_DOWN;
    input.x = x;
    input.y = y;
    switch (button) {
    case GLUT_LEFT_BUTTON: input.button = WorkspacePointerEvent::Button::Left; break;
    case GLUT_MIDDLE_BUTTON: input.button = WorkspacePointerEvent::Button::Middle; break;
    case GLUT_RIGHT_BUTTON: input.button = WorkspacePointerEvent::Button::Right; break;
    case 3: input.button = WorkspacePointerEvent::Button::WheelUp; break;
    case 4: input.button = WorkspacePointerEvent::Button::WheelDown; break;
    default: break;
    }
    return input;
}

WorkspacePointerEvent TheArbiter::translateMouseMotion(
    int x, int y, int dx, int dy) const {
    WorkspacePointerEvent input;
    input.type = WorkspacePointerEvent::Type::Motion;
    input.button = WorkspacePointerEvent::Button::Left;
    input.pressed = true;
    input.x = x;
    input.y = y;
    input.dx = dx;
    input.dy = dy;
    return input;
}

void TheArbiter::cycleMulphyWorkspace(int direction) {

    if (direction == 0) return;

    static constexpr WorkspaceId order[] = {

        WorkspaceId::PARTICLE_SIM,
        WorkspaceId::ATOMIC_PARTICLES,
        WorkspaceId::MULTIPHYSICS_SIM
    };

    constexpr int count =
        static_cast<int>(sizeof(order) / sizeof(order[0]));

    int currentIndex = 0;

    for (int i = 0; i < count; i++) {

        if (order[i] == m_navigation.workspace) {
            currentIndex = i;
            break;
        }
    }

    const int step =
        direction < 0 
        ? -1 
        : +1;

    const int nextIndex =
        (currentIndex + step + count) % count;

    m_navigation.workspace = order[nextIndex];
}

void TheArbiter::requestEnterDomain(WorkspaceDomain domain) {
    m_navigationRequest.type = NavigationRequestType::ENTER_DOMAIN;
    m_navigationRequest.domain = domain;
}

void TheArbiter::requestReturnToGlobalShell(WorkspaceDomain domain) {
    m_navigationRequest.type = NavigationRequestType::RETURN_GLOBAL_SHELL;
    m_navigationRequest.domain = domain;
}

TheArbiter::NavigationRequest TheArbiter::takeNavigationRequest() {
    const NavigationRequest request = m_navigationRequest;
    m_navigationRequest = NavigationRequest{};
    return request;
}

bool TheArbiter::isGlobalShell() const {
    return m_navigation.layer == ApplicationLayer::GLOBAL_SHELL;
}

bool TheArbiter::isDomainSelection() const {
    return m_navigation.layer == ApplicationLayer::DOMAIN_SELECTION;
}

bool TheArbiter::isWorkspaceLayer() const {
    return m_navigation.layer == ApplicationLayer::DOMAIN_SELECTION ||
        m_navigation.layer == ApplicationLayer::WORKSPACE_CONFIGURATION ||
        m_navigation.layer == ApplicationLayer::ACTIVE_WORKSPACE;
}
