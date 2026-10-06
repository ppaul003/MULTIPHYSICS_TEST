#ifndef NDMSM_VIEWPORT_EM_H
#define NDMSM_VIEWPORT_EM_H

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <algorithm>

#include "WorkspacePresentationEM.h"

class ViewPort {
public:
    ViewPort() = default;

    void resize(int width, int height);
    void applyPerspective(float fovDegrees = 60.0f);
    void drawOverlay(const WorkspacePresentation& presentation);

    int getWidth() const { return m_windowWidth; }
    int getHeight() const { return m_windowHeight; }
    float getAspect() const;

private:
    static int clampPositive(int value) { return (std::max)(1, value); }

    void beginOverlay2D();
    void endOverlay2D();
    void drawText2D(float x, float y, const char* text, void* font);

    void updatePanelAnimation(bool visible);
    void updateSubLayerPanelAnimation(bool visible);
    float panelOffsetX() const;
    float panelX(float x) const { return x + panelOffsetX(); }

    void drawWorkspaceFrame(const WorkspacePresentation& presentation);
    void drawRuntimeStatusPanel(const WorkspaceRuntimeStatus& status);
    void drawSubLayerPresentation(const WorkspacePresentation& presentation);
    void drawPanelBackground();
    void drawHeader(const WorkspacePresentation& presentation);
    void drawSections(const WorkspacePresentation& presentation);
    void drawFooter(const WorkspacePresentation& presentation);

private:
    int m_windowWidth = 1920;
    int m_windowHeight = 1080;
    float m_fov = 60.0f;

    float m_panelWidth = 650.0f;
    float m_margin = 42.0f;
    float m_rowSpacing = 55.0f;
    float m_panelSlide = 1.0f;
    float m_subLayerPanelSlide = 0.0f;
};

#endif
