#include "CameraEM.h"

#include <GL/glew.h>
#include <GL/wglew.h>
#include <GL/freeglut.h>

using namespace std;

namespace {
    constexpr float kMenuX = 1.65f;
    constexpr float kMenuY = 0.00f;
    constexpr float kMenuZ = -8.00f;
    constexpr float kMenuPitch = 18.0f;

    constexpr float kStandard3DZ = -5.00f;

    constexpr float kStandard2DX = 1.40f;
    constexpr float kStandard2DY = 0.00f;
    constexpr float kStandard2DZ = -6.00f;
    constexpr float kPreMenu2DZ = -8.00f;
    constexpr float kDegreesToRadians = 0.01745329251994329577f;
    constexpr float kFreeMoveSpeed = 2.0f;
    constexpr float kFreeLookDegreesPerPixel = 0.20f;

    void multiplyRotation(const float* left, const float* right, float* result) {
        float product[9]{};
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                for (int k = 0; k < 3; ++k) {
                    product[row * 3 + col] += left[row * 3 + k] * right[k * 3 + col];
                }
            }
        }
        std::copy(product, product + 9, result);
    }

    void makeViewRotation(const float* angles, float* result) {
        const float cx = std::cos(angles[0] * kDegreesToRadians);
        const float sx = std::sin(angles[0] * kDegreesToRadians);
        const float cy = std::cos(angles[1] * kDegreesToRadians);
        const float sy = std::sin(angles[1] * kDegreesToRadians);
        const float cz = std::cos(angles[2] * kDegreesToRadians);
        const float sz = std::sin(angles[2] * kDegreesToRadians);
        const float rx[9]{1, 0, 0, 0, cx, -sx, 0, sx, cx};
        const float ry[9]{cy, 0, sy, 0, 1, 0, -sy, 0, cy};
        const float rz[9]{cz, -sz, 0, sz, cz, 0, 0, 0, 1};
        multiplyRotation(rx, ry, result);
        multiplyRotation(result, rz, result);
    }

    void normalizeAxis(float* axis) {
        const float length = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
        if (length > 0.0f) {
            for (int c = 0; c < 3; ++c) axis[c] /= length;
        }
    }

    void orthonormalizeRotation(float* rotation) {
        normalizeAxis(rotation);
        const float projection = rotation[0] * rotation[3] + rotation[1] * rotation[4] + rotation[2] * rotation[5];
        for (int c = 0; c < 3; ++c) rotation[3 + c] -= projection * rotation[c];
        normalizeAxis(rotation + 3);
        rotation[6] = rotation[1] * rotation[5] - rotation[2] * rotation[4];
        rotation[7] = rotation[2] * rotation[3] - rotation[0] * rotation[5];
        rotation[8] = rotation[0] * rotation[4] - rotation[1] * rotation[3];
    }

    float smoothStep01(float t) {
        if (t <= 0.0f) return 0.0f;
        if (t >= 1.0f) return 1.0f;
        return t * t * (3.0f - 2.0f * t);
    }

    float lerp(float a, float b, float t) {
        return a + (b - a) * t;
    }
}

CameraProcessor::CameraProcessor() :
    m_cameraTrans{ kMenuX, kMenuY, kMenuZ },
    m_cameraRot{ kMenuPitch, 0.0f, 0.0f },
    m_cameraTransLag{ kMenuX, kMenuY, kMenuZ },
    m_cameraRotLag{ kMenuPitch, 0.0f, 0.0f } {
}

void CameraProcessor::updateLag() {
    if (m_poseTransitionActive || m_freeViewActive) return;

    for (int c = 0; c < 3; c++) {
        m_cameraTransLag[c] +=
            (m_cameraTrans[c] - m_cameraTransLag[c]) * kInertia;
        m_cameraRotLag[c] +=
            (m_cameraRot[c] - m_cameraRotLag[c]) * kInertia;
    }
}

void CameraProcessor::applyCameraTransform() {
    float view[16];
    buildViewMatrix(view);
    glMultMatrixf(view);
}

void CameraProcessor::buildViewMatrix(float* view) const {
    std::fill(view, view + 16, 0.0f);
    view[15] = 1.0f;
    if (m_freeViewActive) {
        // OpenGL expects column-major storage for the view matrix [ R | -R * eye ].
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                view[col * 4 + row] = m_freeRotation[row * 3 + col];
                view[12 + row] -= m_freeRotation[row * 3 + col] * m_freeEye[col];
            }
        }
        return;
    }

    // Same T * Rx * Ry * Rz transform as the original OpenGL orbit path.
    float rotation[9];
    makeViewRotation(m_cameraRotLag, rotation);
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col)
            view[col * 4 + row] = rotation[row * 3 + col];
        view[12 + row] = m_cameraTransLag[row];
    }
}

bool CameraProcessor::getCenterViewRay(
    glm::vec3& originWorld, glm::vec3& directionWorld) const {
    float view[16];
    buildViewMatrix(view);

    // Inverse rigid transform: eye = R^T * -translation and forward = R^T * -Z.
    // This is the ixformPoint/ixform convention; direction excludes translation.
    glm::vec3 origin;
    glm::vec3 direction;
    for (int c = 0; c < 3; ++c) {
        origin[c] = -(view[c * 4] * view[12] +
            view[c * 4 + 1] * view[13] + view[c * 4 + 2] * view[14]);
        direction[c] = -view[c * 4 + 2];
        if (!std::isfinite(origin[c]) || !std::isfinite(direction[c])) return false;
    }
    const float length = std::sqrt(direction.x * direction.x +
        direction.y * direction.y + direction.z * direction.z);
    if (!std::isfinite(length) || length <= 0.0f) return false;

    originWorld = origin;
    directionWorld = direction / length;
    return true;
}

void CameraProcessor::setMenuPoseTarget() {
    m_cameraTrans[0] = kMenuX;
    m_cameraTrans[1] = kMenuY;
    m_cameraTrans[2] = kMenuZ;
    m_cameraRot[0] = kMenuPitch;
    m_cameraRot[1] = 0.0f;
    m_cameraRot[2] = 0.0f;
}

void CameraProcessor::setStandard3DTarget() {
    m_cameraTrans[0] = 0.0f;
    m_cameraTrans[1] = 0.0f;
    m_cameraTrans[2] = kStandard3DZ;
    m_cameraRot[0] = 0.0f;
    m_cameraRot[1] = 0.0f;
    m_cameraRot[2] = 0.0f;
}

void CameraProcessor::setStandard2DTarget() {
    m_cameraTrans[0] = kStandard2DX;
    m_cameraTrans[1] = kStandard2DY;
    m_cameraTrans[2] = kStandard2DZ;
    m_cameraRot[0] = 0.0f;
    m_cameraRot[1] = 0.0f;
    m_cameraRot[2] = 0.0f;
}

void CameraProcessor::setBehaviorMode(CameraBehaviorMode mode) {
    endFreeView();
    m_behaviorMode = mode;

    switch (mode) {
    case CAM_STANDARD_3D:
        setStandard3DTarget();
        break;
    case CAM_STANDARD_2D:
        setStandard2DTarget();
        break;
    case CAM_MENU_PREVIEW:
    default:
        setMenuPoseTarget();
        break;
    }
}

void CameraProcessor::beginTransitionToPose(
    float tx, float ty, float tz,
    float rx, float ry, float rz,
    float duration) {

    endFreeView();
    for (int c = 0; c < 3; c++) {
        m_poseStartTrans[c] = m_cameraTransLag[c];
        m_poseStartRot[c] = m_cameraRotLag[c];
        m_cameraTrans[c] = m_poseStartTrans[c];
        m_cameraRot[c] = m_poseStartRot[c];
    }

    m_poseTargetTrans[0] = tx;
    m_poseTargetTrans[1] = ty;
    m_poseTargetTrans[2] = tz;
    m_poseTargetRot[0] = rx;
    m_poseTargetRot[1] = ry;
    m_poseTargetRot[2] = rz;

    m_poseTransitionElapsed = 0.0f;
    m_poseTransitionDuration = duration;

    if (duration <= 0.0f) {
        for (int c = 0; c < 3; c++) {
            m_cameraTrans[c] = m_poseTargetTrans[c];
            m_cameraRot[c] = m_poseTargetRot[c];
            m_cameraTransLag[c] = m_poseTargetTrans[c];
            m_cameraRotLag[c] = m_poseTargetRot[c];
        }
        m_poseTransitionActive = false;
        return;
    }

    m_poseTransitionActive = true;
}

void CameraProcessor::beginTransitionToStandard3D(float duration) {
    beginTransitionToPose(
        0.0f, 0.0f, kStandard3DZ,
        0.0f, 0.0f, 0.0f,
        duration
    );
}

void CameraProcessor::beginTransitionToCentered2D(float duration) {
    beginTransitionToPose(
        0.0f, 0.0f, kStandard2DZ,
        0.0f, 0.0f, 0.0f,
        duration);
}

void CameraProcessor::beginTransitionToStandard2D(float duration) {
    beginTransitionToPose(
        kStandard2DX, kStandard2DY, kStandard2DZ,
        0.0f, 0.0f, 0.0f,
        duration
    );
}

void CameraProcessor::beginTransitionToPreMenu2D(float duration) {
    beginTransitionToPose(
        0.0f, 0.0f, kPreMenu2DZ,
        0.0f, 0.0f, 0.0f,
        duration
    );
}

void CameraProcessor::beginTransitionToMenu(float duration) {
    beginTransitionToPose(
        kMenuX, kMenuY, kMenuZ,
        kMenuPitch, 0.0f, 0.0f,
        duration
    );
}

void CameraProcessor::updatePoseTransition(float deltaTime) {
    if (!m_poseTransitionActive) return;
    if (deltaTime < 0.0f) deltaTime = 0.0f;

    m_poseTransitionElapsed += deltaTime;

    if (m_poseTransitionDuration <= 0.0f) {
        m_poseTransitionActive = false;
        return;
    }

    const float rawT = m_poseTransitionElapsed / m_poseTransitionDuration;
    const float t = smoothStep01(rawT);

    for (int c = 0; c < 3; c++) {
        const float translation = lerp(
            m_poseStartTrans[c], m_poseTargetTrans[c], t);
        const float rotation = lerp(
            m_poseStartRot[c], m_poseTargetRot[c], t);

        m_cameraTrans[c] = translation;
        m_cameraTransLag[c] = translation;
        m_cameraRot[c] = rotation;
        m_cameraRotLag[c] = rotation;
    }

    if (rawT >= 1.0f) {
        for (int c = 0; c < 3; c++) {
            m_cameraTrans[c] = m_poseTargetTrans[c];
            m_cameraTransLag[c] = m_poseTargetTrans[c];
            m_cameraRot[c] = m_poseTargetRot[c];
            m_cameraRotLag[c] = m_poseTargetRot[c];
        }
        m_poseTransitionElapsed = m_poseTransitionDuration;
        m_poseTransitionActive = false;
    }
}

void CameraProcessor::orbit(float dx, float dy) {
    if (!orbitEnabled()) return;

    // Mouse Y -> pitch
    m_cameraRot[0] += dy / 5.0f;

    // Mouse X -> yaw
    m_cameraRot[1] += dx / 5.0f;
}

void CameraProcessor::zoom(float amount) {
    if (!zoomEnabled()) return;

    const float distance = max(1.0f, fabs(m_cameraTrans[2]));

    m_cameraTrans[2] += amount * distance;
    m_cameraTrans[2] = clamp(m_cameraTrans[2], -30.0f, -1.5f);
}

void CameraProcessor::beginFreeView() {
    if (m_freeViewActive || !orbitEnabled()) return;

    // Capture the displayed pose, not the still-converging orbit target.
    for (int c = 0; c < 3; ++c) {
        m_savedOrbitTrans[c] = m_cameraTransLag[c];
        m_savedOrbitRot[c] = m_cameraRotLag[c];
    }
    makeViewRotation(m_savedOrbitRot, m_freeRotation);
    for (int c = 0; c < 3; ++c) {
        m_freeEye[c] = 0.0f;
        for (int row = 0; row < 3; ++row) {
            m_freeEye[c] -= m_freeRotation[row * 3 + c] * m_savedOrbitTrans[row];
        }
    }
    m_freeViewActive = true;
}

void CameraProcessor::endFreeView() {
    if (!m_freeViewActive) return;

    for (int c = 0; c < 3; ++c) {
        m_cameraTrans[c] = m_cameraTransLag[c] = m_savedOrbitTrans[c];
        m_cameraRot[c] = m_cameraRotLag[c] = m_savedOrbitRot[c];
        m_savedOrbitTrans[c] = m_savedOrbitRot[c] = m_freeEye[c] = 0.0f;
    }
    std::fill(m_freeRotation, m_freeRotation + 9, 0.0f);
    m_freeViewActive = false;
}

void CameraProcessor::moveFree(float forward, float right, float deltaTime) {
    if (!m_freeViewActive || deltaTime <= 0.0f) return;

    // Diagonal movement has the same speed as movement along one local axis.
    const float inputLength = std::sqrt(forward * forward + right * right);
    const float distance = kFreeMoveSpeed * deltaTime / (std::max)(1.0f, inputLength);
    for (int c = 0; c < 3; ++c) {
        m_freeEye[c] += (right * m_freeRotation[c] - forward * m_freeRotation[6 + c]) * distance;
    }
}

void CameraProcessor::lookFree(float dx, float dy) {
    if (!m_freeViewActive || (dx == 0.0f && dy == 0.0f)) return;

    // Rotate around the current camera axes without moving the eye. This also
    // preserves arbitrary orbit orientation on entry, including past vertical.
    const float angles[3]{dy * kFreeLookDegreesPerPixel, dx * kFreeLookDegreesPerPixel, 0.0f};
    float localRotation[9];
    makeViewRotation(angles, localRotation);
    multiplyRotation(localRotation, m_freeRotation, m_freeRotation);
    orthonormalizeRotation(m_freeRotation);
}
