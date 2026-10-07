#ifndef NDMSM_CAMERA_EM_H
#define NDMSM_CAMERA_EM_H

#include <algorithm>
#include <cmath>
#include <glm/vec3.hpp>


class CameraProcessor {
public:
    enum CameraBehaviorMode {
        CAM_MENU_PREVIEW = 0,
        CAM_STANDARD_3D,
        CAM_STANDARD_2D
    };

    CameraProcessor();

    void updateLag();
    void applyCameraTransform();
    // The displayed (lagged orbit or free-view) camera pose, independent of GL state.
    bool getCenterViewRay(glm::vec3& originWorld, glm::vec3& directionWorld) const;

    void setBehaviorMode(CameraBehaviorMode mode, float distanceScale = 1.0f);
    CameraBehaviorMode getBehaviorMode() const { return m_behaviorMode; }

    void beginTransitionToStandard3D(float duration = 0.75f, float distanceScale = 1.0f);
    void beginTransitionToCentered2D(float duration = 0.45f);
    void beginTransitionToStandard2D(float duration = 0.30f);
    void beginTransitionToPreMenu2D(float duration = 0.45f);
    void beginTransitionToMenu(float duration = 0.75f, float distanceScale = 1.0f);
    void beginRelativeDistanceScale(float scale, float duration);

    void updatePoseTransition(float deltaTime);
    bool poseTransitionActive() const { return m_poseTransitionActive; }

    void orbit(float dx, float dy);
    void zoom(float amount);

    void beginFreeView();
    void endFreeView();
    bool freeViewActive() const { return m_freeViewActive; }
    void moveFree(float forward, float right, float deltaTime);
    void lookFree(float dx, float dy);

    bool orbitEnabled() const { return !m_freeViewActive && !m_poseTransitionActive && m_behaviorMode == CAM_STANDARD_3D; }
    bool zoomEnabled() const { return !m_freeViewActive && !m_poseTransitionActive && (m_behaviorMode == CAM_STANDARD_3D || m_behaviorMode == CAM_STANDARD_2D); }

private:
    void buildViewMatrix(float* view) const;
    void beginTransitionToPose(
        float tx, float ty, float tz,
        float rx, float ry, float rz,
        float duration);

    void setMenuPoseTarget();
    void setStandard3DTarget();
    void setStandard2DTarget();

private:
    static constexpr float kInertia = 0.10f;

    CameraBehaviorMode m_behaviorMode = CAM_MENU_PREVIEW;

    bool m_poseTransitionActive = false;
    float m_poseTransitionElapsed = 0.0f;
    float m_poseTransitionDuration = 0.75f;

    float m_poseStartTrans[3]{};
    float m_poseStartRot[3]{};
    float m_poseTargetTrans[3]{};
    float m_poseTargetRot[3]{};

    float m_cameraTrans[3];
    float m_cameraRot[3];
    float m_cameraTransLag[3];
    float m_cameraRotLag[3];

    bool m_freeViewActive = false;
    float m_savedOrbitTrans[3]{};
    float m_savedOrbitRot[3]{};
    float m_freeEye[3]{};
    // Row-major world-to-camera rotation; its rows are the camera's local axes.
    float m_freeRotation[9]{};
};

#endif
