#ifndef NDMSM_INTERACTIONS_EM_H
#define NDMSM_INTERACTIONS_EM_H

#include <GL/glew.h>
#include <GL/wglew.h>
#include <GL/freeglut.h>

#include <cctype>
#include <cmath>


class KeyboardInput {
public:
    enum KeySignal {
        KEY_NONE = 0,
        KEY_ESCAPE,
        KEY_ENTER,
        KEY_TAB,
        KEY_SPACE,
        KEY_A,
        KEY_D,
        KEY_E,
        KEY_Q,
        KEY_W,
        KEY_S,
        KEY_UNKNOWN
    };

    struct KeyEvent {
        KeySignal signal = KEY_NONE;
        unsigned char rawKey = 0;
        int x = 0;
        int y = 0;
    };

    KeyEvent onKey(unsigned char key, int x, int y) const {
        KeyEvent event;
        event.rawKey = key;
        event.x = x;
        event.y = y;
        event.signal = decode(key);
        return event;
    }

private:
    KeySignal decode(unsigned char key) const {
        if (key == 27) return KEY_ESCAPE;
        if (key == 13) return KEY_ENTER;
        if (key == '\t') return KEY_TAB;
        if (key == ' ') return KEY_SPACE;

        const unsigned char lowered = static_cast<unsigned char>(
            std::tolower(static_cast<unsigned char>(key)));

        switch (lowered) {
        case 'a': return KEY_A;
        case 'd': return KEY_D;
        case 'e': return KEY_E;
        case 'q': return KEY_Q;
        case 'w': return KEY_W;
        case 's': return KEY_S;
        default:  return KEY_UNKNOWN;
        }
    }
};


class MouseInput {
public:
    void onButton(
        int button,
        int state,
        int x,
        int y) {

        if (button != GLUT_LEFT_BUTTON)
            return;

        m_leftDown =
            state == GLUT_DOWN;

        m_lastX = x;
        m_lastY = y;
    }

    bool onMotion(
        int x,
        int y,
        int& dx,
        int& dy) {

        dx = x - m_lastX;
        dy = y - m_lastY;

        m_lastX = x;
        m_lastY = y;

        return m_leftDown;
    }

private:
    bool m_leftDown = false;

    int m_lastX = 0;
    int m_lastY = 0;
};
#endif
