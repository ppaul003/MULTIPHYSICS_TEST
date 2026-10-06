#include "ViewPortEM.h"

#include <chrono>
#include <cmath>
#include <string>

using namespace std;

void ViewPort::resize(int width, int height) {
    m_windowWidth = clampPositive(width);
    m_windowHeight = clampPositive(height);
    glViewport(0, 0, m_windowWidth, m_windowHeight);
}

void ViewPort::applyPerspective(float fovDegrees) {
    m_fov = fovDegrees;
    glViewport(0, 0, m_windowWidth, m_windowHeight);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(m_fov, getAspect(), 0.01, 100.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

float ViewPort::getAspect() const {
    if (m_windowHeight <= 0) return 1.0f;
    return static_cast<float>(m_windowWidth) /
        static_cast<float>(m_windowHeight);
}

void ViewPort::beginOverlay2D() {
    glViewport(0, 0, m_windowWidth, m_windowHeight);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0.0, static_cast<double>(m_windowWidth),
        static_cast<double>(m_windowHeight), 0.0);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
}

void ViewPort::endOverlay2D() {
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
}

void ViewPort::drawText2D(float x, float y, const char* text, void* font) {
    if (!text) return;
    glRasterPos2f(x, y);
    for (const char* p = text; *p; ++p)
        glutBitmapCharacter(font, *p);
}

void ViewPort::updatePanelAnimation(bool visible) {
    const float target = visible ? 1.0f : 0.0f;
    m_panelSlide += (target - m_panelSlide) * 0.15f;
    if (m_panelSlide < 0.001f) m_panelSlide = 0.0f;
    if (m_panelSlide > 0.999f) m_panelSlide = 1.0f;
}

void ViewPort::updateSubLayerPanelAnimation(bool visible) {
    const float target = visible ? 1.0f : 0.0f;
    m_subLayerPanelSlide +=
        (target - m_subLayerPanelSlide) * 0.18f;

    if (m_subLayerPanelSlide < 0.001f)
        m_subLayerPanelSlide = 0.0f;

    if (m_subLayerPanelSlide > 0.999f)
        m_subLayerPanelSlide = 1.0f;
}

float ViewPort::panelOffsetX() const {
    const float hiddenX = -(m_panelWidth + m_margin + 24.0f);
    return hiddenX * (1.0f - m_panelSlide);
}

void ViewPort::drawWorkspaceFrame(const WorkspacePresentation& p) {
    const float margin = 24.0f;
    const float x0 = margin;
    const float y0 = margin;
    const float x1 = static_cast<float>(m_windowWidth) - margin;
    const float y1 = static_cast<float>(m_windowHeight) - margin;

    float r = 0.85f, g = 0.95f, b = 1.00f, alpha = 0.30f;

    if (p.frameTone == WorkspaceStatusTone::Transition) {
        r = 1.00f; g = 0.65f; b = 0.15f; alpha = 0.95f;
    }
    else if (p.frameTone == WorkspaceStatusTone::Ready) {
        r = 0.45f; g = 0.95f; b = 1.00f; alpha = 0.85f;
    }

    if (p.frameBlink) {
        using Clock = chrono::steady_clock;
        const float seconds = chrono::duration<float>(
            Clock::now().time_since_epoch()).count();
        if (fmod(seconds, 0.50f) >= 0.25f) alpha *= 0.20f;
    }

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_TEXTURE_2D);
    glUseProgram(0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glLineWidth(1.0f);
    glColor4f(r, g, b, alpha);
    glBegin(GL_LINE_LOOP);
    glVertex2f(x0, y0);
    glVertex2f(x1, y0);
    glVertex2f(x1, y1);
    glVertex2f(x0, y1);
    glEnd();
}

void ViewPort::drawPanelBackground() {
    const float x0 = panelX(m_margin);
    const float y0 = m_margin;
    const float x1 = panelX(m_panelWidth);
    const float y1 = static_cast<float>(m_windowHeight) - m_margin;

    glColor4f(0.02f, 0.04f, 0.06f, 0.76f * m_panelSlide);
    glBegin(GL_QUADS);
    glVertex2f(x0, y0); glVertex2f(x1, y0);
    glVertex2f(x1, y1); glVertex2f(x0, y1);
    glEnd();

    glLineWidth(1.5f);
    glColor4f(1.0f, 1.0f, 1.0f, 0.95f * m_panelSlide);
    glBegin(GL_LINE_LOOP);
    glVertex2f(x0, y0); glVertex2f(x1, y0);
    glVertex2f(x1, y1); glVertex2f(x0, y1);
    glEnd();
    glLineWidth(1.0f);
}

void ViewPort::drawHeader(const WorkspacePresentation& p) {
    const float x = panelX(95.0f);

    glColor4f(
        1.0f,
        1.0f,
        1.0f,
        m_panelSlide
    );

    drawText2D(
        x,
        100.0f,
        "ANAHEIM SYSTEMS DYNAMICS",
        GLUT_BITMAP_HELVETICA_18
    );

    drawText2D(
        x,
        128.0f,
        "Simulation Module Test & Debug",
        GLUT_BITMAP_HELVETICA_18
    );

    drawText2D(
        x,
        156.0f,
        "Development Ver. TEST",
        GLUT_BITMAP_HELVETICA_18
    );

    // ----------------------------------------
    // Workspace / Layer information
    // ----------------------------------------
    if (!p.workspaceName.empty()) {
        glColor4f(
            0.82f,
            0.86f,
            0.90f,
            m_panelSlide
        );

        drawText2D(
            x,
            190.0f,
            p.workspaceName.c_str(),
            GLUT_BITMAP_HELVETICA_18
        );
    }

    // ----------------------------------------
    // Divider
    // ----------------------------------------
    glColor4f(
        0.55f,
        0.60f,
        0.65f,
        0.65f * m_panelSlide
    );

    glLineWidth(1.0f);

    glBegin(GL_LINES);
    glVertex2f(x, 210.0f);
    glVertex2f(panelX(505.0f), 210.0f);
    glEnd();

    // ----------------------------------------
    // Optional layer label
    // ----------------------------------------
    if (!p.layerLabel.empty()) {
        glColor4f(
            0.72f,
            0.78f,
            0.82f,
            m_panelSlide
        );

        drawText2D(
            x,
            232.0f,
            p.layerLabel.c_str(),
            GLUT_BITMAP_HELVETICA_18
        );
    }
}

void ViewPort::drawSections(const WorkspacePresentation& p) {
    float y = 290.0f;
    const float x = 95.0f;

    for (const WorkspacePanelSection& section : p.sections) {
        if (!section.heading.empty()) {
            glColor4f(0.85f, 0.95f, 1.0f, m_panelSlide);
            drawText2D(panelX(x), y, section.heading.c_str(), GLUT_BITMAP_HELVETICA_18);
            y += 42.0f;
        }

        for (const WorkspacePanelRow& row : section.rows) {
            if (row.selected) {
                glColor4f(0.45f, 1.0f, 0.65f, m_panelSlide);
                drawText2D(panelX(x - 28.0f), y, ">", GLUT_BITMAP_HELVETICA_18);
            }
            else {
                glColor4f(0.72f, 0.78f, 0.82f, m_panelSlide);
            }

            std::string line = row.label;
            if (!row.value.empty()) {
                line += " { ";
                line += row.value;
                line += " }";
            }
            drawText2D(panelX(x), y, line.c_str(), GLUT_BITMAP_HELVETICA_18);
            y += m_rowSpacing;
        }
    }

    if (!p.statusLine.empty()) {
        y += 12.0f;
        float alpha = m_panelSlide;

        if (p.statusBlink) {
            using Clock = chrono::steady_clock;
            const float seconds = chrono::duration<float>(
                Clock::now().time_since_epoch()).count();
            if (fmod(seconds, 0.50f) >= 0.25f) alpha *= 0.20f;
        }

        switch (p.statusTone) {
        case WorkspaceStatusTone::Ready:
            glColor4f(0.45f, 1.0f, 0.65f, alpha);
            break;
        case WorkspaceStatusTone::Warning:
            glColor4f(1.0f, 0.45f, 0.45f, alpha);
            break;
        case WorkspaceStatusTone::Transition:
            glColor4f(1.0f, 0.65f, 0.15f, alpha);
            break;
        default:
            glColor4f(0.72f, 0.78f, 0.82f, alpha);
            break;
        }

        drawText2D(
            panelX(x),
            y,
            p.statusLine.c_str(),
            GLUT_BITMAP_HELVETICA_18
        );

        y += 44.0f;
    }

    // ---------------------------------------------------------
    // Informational lines between status and footer.
    // ---------------------------------------------------------
    for (const string& line : p.postStatusLines) {

        glColor4f(0.72f, 0.78f, 0.82f, m_panelSlide);

        drawText2D(
            panelX(x),
            y,
            line.c_str(),
            GLUT_BITMAP_HELVETICA_18
        );

        y += 34.0f;
    }
    for (const auto& line : p.capabilityLines) {
        switch (line.tone) {
        case WorkspaceStatusTone::Ready: glColor4f(0.45f, 1.0f, 0.65f, m_panelSlide); break;
        case WorkspaceStatusTone::Warning: glColor4f(1.0f, 0.45f, 0.45f, m_panelSlide); break;
        case WorkspaceStatusTone::Transition: glColor4f(1.0f, 0.65f, 0.15f, m_panelSlide); break;
        default: glColor4f(0.72f, 0.78f, 0.82f, m_panelSlide); break;
        }
        drawText2D(panelX(x), y, line.text.c_str(), GLUT_BITMAP_HELVETICA_18);
        y += 34.0f;
    }
}

void ViewPort::drawRuntimeStatusPanel(
    const WorkspaceRuntimeStatus& status) {

    if (!status.visible) return;

    const float panelLeft = 24.0f;
    const float panelTop = 24.0f;
    const float panelBottom = 184.0f + 22.0f * static_cast<float>(status.detailLines.size());

    const float viewportRight =
        (std::max)(panelLeft + 640.0f,
            static_cast<float>(m_windowWidth) - 24.0f);

    const float normalPanelRight =
        (std::min)(1120.0f, viewportRight);

    const float panelRight = normalPanelRight;

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_TEXTURE_2D);
    glUseProgram(0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glColor4f(0.02f, 0.04f, 0.06f, 0.55f);
    glBegin(GL_QUADS);
    glVertex2f(panelLeft, panelTop);
    glVertex2f(panelRight, panelTop);
    glVertex2f(panelRight, panelBottom);
    glVertex2f(panelLeft, panelBottom);
    glEnd();

    glColor3f(0.85f, 0.95f, 1.0f);
    drawText2D(
        40.0f,
        52.0f,
        status.titleLine.c_str(),
        GLUT_BITMAP_HELVETICA_18
    );
    drawText2D(
        40.0f,
        82.0f,
        status.contextLine.c_str(),
        GLUT_BITMAP_HELVETICA_18
    );

    switch (status.objectTone) {
    case WorkspaceStatusTone::Ready:
        glColor3f(0.45f, 1.0f, 0.65f);
        break;
    case WorkspaceStatusTone::Warning:
        glColor3f(1.0f, 0.45f, 0.45f);
        break;
    case WorkspaceStatusTone::Transition:
        glColor3f(1.0f, 0.65f, 0.15f);
        break;
    case WorkspaceStatusTone::Neutral:
    default:
        glColor3f(0.72f, 0.78f, 0.82f);
        break;
    }

    drawText2D(
        40.0f,
        108.0f,
        status.objectLine.c_str(),
        GLUT_BITMAP_HELVETICA_18
    );

    glColor3f(0.85f, 0.95f, 1.0f);
    drawText2D(
        40.0f,
        134.0f,
        status.helpLine.c_str(),
        GLUT_BITMAP_HELVETICA_18
    );
    float detailY = 162.0f;
    for (const std::string& line : status.detailLines) {
        drawText2D(40.0f, detailY, line.c_str(), GLUT_BITMAP_HELVETICA_12);
        detailY += 22.0f;
    }
}

void ViewPort::drawSubLayerPresentation(
    const WorkspacePresentation& presentation) {

    const float alpha = m_subLayerPanelSlide;
    const float panelWidth = 650.0f;
    const float x0 = m_margin;
    const float y0Visible = 170.0f;

    const float panelHeight =
        static_cast<float>(m_windowHeight) -
        y0Visible - m_margin;

    const float hiddenOffsetY =
        panelHeight + m_margin + 24.0f;

    const float y0 = y0Visible +
        hiddenOffsetY * (1.0f - alpha);

    const float x1 = x0 + panelWidth;
    const float y1 = y0 + panelHeight;

    const float sectionX = x0 + 54.0f;
    const float labelX = x0 + 84.0f;
    const float dividerX0 = x0 + 42.0f;
    const float dividerX1 = x1 - 42.0f;

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_TEXTURE_2D);
    glUseProgram(0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glColor4f(0.02f, 0.04f, 0.06f, 0.80f * alpha);
    glBegin(GL_QUADS);
    glVertex2f(x0, y0);
    glVertex2f(x1, y0);
    glVertex2f(x1, y1);
    glVertex2f(x0, y1);
    glEnd();

    glLineWidth(1.5f);
    glColor4f(1.0f, 1.0f, 1.0f, 0.95f * alpha);
    glBegin(GL_LINE_LOOP);
    glVertex2f(x0, y0);
    glVertex2f(x1, y0);
    glVertex2f(x1, y1);
    glVertex2f(x0, y1);
    glEnd();

    glColor4f(0.85f, 0.95f, 1.0f, alpha);
    drawText2D(
        sectionX,
        y0 + 40.0f,
        presentation.workspaceName.c_str(),
        GLUT_BITMAP_HELVETICA_18
    );
    drawText2D(
        sectionX,
        y0 + 70.0f,
        presentation.subLayerLabel.c_str(),
        GLUT_BITMAP_HELVETICA_18
    );

    const auto drawDivider = [&](float y) {
        glColor4f(0.85f, 0.95f, 1.0f, 0.45f * alpha);
        glBegin(GL_LINES);
        glVertex2f(dividerX0, y);
        glVertex2f(dividerX1, y);
        glEnd();
    };

    const auto drawSubRow = [&](float y, const WorkspacePanelRow& row) {
        if (row.selected) {
            glColor4f(0.45f, 1.0f, 0.65f, alpha);
            drawText2D(
                labelX - 22.0f,
                y,
                ">",
                GLUT_BITMAP_HELVETICA_18
            );
        }
        else {
            glColor4f(0.72f, 0.78f, 0.82f, alpha);
        }

        string line = row.label;
        if (!row.value.empty()) {
            line += " { ";
            line += row.value;
            line += " }";
        }

        drawText2D(
            labelX,
            y,
            line.c_str(),
            GLUT_BITMAP_HELVETICA_18
        );
    };

    drawDivider(y0 + 105.0f);
    float y = y0 + 145.0f;

    for (const WorkspacePanelSection& section : presentation.sections) {
        if (y >= y1 - 145.0f) break;

        glColor4f(0.85f, 0.95f, 1.0f, alpha);
        drawText2D(
            sectionX,
            y,
            section.heading.c_str(),
            GLUT_BITMAP_HELVETICA_18
        );
        y += 42.0f;

        for (const WorkspacePanelRow& row : section.rows) {
            if (y >= y1 - 145.0f) break;
            drawSubRow(y, row);
            y += 42.0f;
        }

        drawDivider(y + 4.0f);
        y += 40.0f;
    }

    drawDivider(y1 - 92.0f);
    glColor4f(0.75f, 0.75f, 0.75f, alpha);
    drawText2D(
        sectionX,
        y1 - 58.0f,
        "W/S: Select    A/D: Change value    "
        "E: Activate    TAB: Hide    Q: Back",
        GLUT_BITMAP_HELVETICA_12
    );

    glLineWidth(1.0f);
}

void ViewPort::drawFooter(const WorkspacePresentation& p) {
    const float x = panelX(95.0f);

    // Bottom edge of side panel
    const float panelBottom = static_cast<float>(m_windowHeight) - m_margin;

    // Footer anchored relative to bottom edge.
    const float footerLine2Y = panelBottom - 22.0f;
    const float footerLine1Y = footerLine2Y - 30.0f;
    const float footerDividerY = footerLine1Y - 26.0f;
    // ----------------------------------------
    // Divider
    // ----------------------------------------
    glColor4f(
        0.45f,
        0.50f,
        0.55f,
        0.55f * m_panelSlide
    );

    glBegin(GL_LINES);
    glVertex2f(x, footerDividerY);
    glVertex2f(panelX(505.0f), footerDividerY);
    glEnd();

    // ----------------------------------------
    // Existing footer controls
    // ----------------------------------------
    if (!p.footerLine1.empty()) {
        glColor4f(
            0.70f,
            0.72f,
            0.75f,
            m_panelSlide
        );

        drawText2D(
            x,
            footerLine1Y,
            p.footerLine1.c_str(),
            GLUT_BITMAP_HELVETICA_12
        );
    }

    if (!p.footerLine2.empty()) {
        drawText2D(
            x,
            footerLine2Y,
            p.footerLine2.c_str(),
            GLUT_BITMAP_HELVETICA_12
        );
    }
}

void ViewPort::drawOverlay(const WorkspacePresentation& p) {
    const bool subLayerPanel =
        p.panelLayout == WorkspacePanelLayout::SubLayer;

    updatePanelAnimation(p.panelVisible && !subLayerPanel);
    updateSubLayerPanelAnimation(p.panelVisible && subLayerPanel);

    beginOverlay2D();
    drawWorkspaceFrame(p);
    drawRuntimeStatusPanel(p.runtimeStatus);

    if (subLayerPanel) {
        if (m_subLayerPanelSlide > 0.0f)
            drawSubLayerPresentation(p);
    }
    else {
        if (m_panelSlide > 0.0f) {
            drawPanelBackground();
            drawHeader(p);
            drawSections(p);
            drawFooter(p);
        }
    }

    endOverlay2D();
}
