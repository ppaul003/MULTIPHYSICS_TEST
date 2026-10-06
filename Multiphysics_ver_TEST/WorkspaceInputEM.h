#ifndef NDMSM_WORKSPACE_INPUT_EM_H
#define NDMSM_WORKSPACE_INPUT_EM_H

enum class WorkspaceInputAction {
    None = 0,
    Previous,
    Next,
    Decrease,
    Increase,
    Activate,
    Back,
    RawKey,
    TogglePanel,
    TogglePause
};

struct WorkspaceInputEvent {
    WorkspaceInputAction action = WorkspaceInputAction::None;
    unsigned char rawKey = 0;
    int x = 0;
    int y = 0;
    bool repeated = false;
};

// Platform-independent pointer data. The host translates GLUT callbacks;
// the cartridge decides whether a widget or a workspace camera consumes them.
struct WorkspacePointerEvent {
    enum class Type { Button, Motion };
    enum class Button { None, Left, Middle, Right, WheelUp, WheelDown };
    Type type = Type::Button;
    Button button = Button::None;
    bool pressed = false;
    int x = 0;
    int y = 0;
    int dx = 0;
    int dy = 0;
};

#endif
