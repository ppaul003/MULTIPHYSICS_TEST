#include <GL/glew.h>

#include "FieldDebugRenderer.h"
#include "VoxelField3D.h"
#include "DebugElectrodynamics.h"
#include "rendererEM_Euclid.h"

#include <algorithm>
#include <cmath>

namespace {
    // Compatibility-profile primitives share the particle camera transform.
    // Shader and render state must survive drawing diagnostics and the HUD.
    class ScopedFieldRenderState {
    public:
        ScopedFieldRenderState() {
            glGetIntegerv(GL_CURRENT_PROGRAM, &m_program);
            glPushAttrib(GL_ENABLE_BIT | GL_CURRENT_BIT | GL_COLOR_BUFFER_BIT |
                GL_DEPTH_BUFFER_BIT | GL_LINE_BIT | GL_POINT_BIT | GL_TEXTURE_BIT);
            glUseProgram(0);
            glDisable(GL_LIGHTING);
            glDisable(GL_TEXTURE_2D);
            glDisable(GL_POINT_SPRITE_ARB);
            glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_LEQUAL);
            glDepthMask(GL_FALSE);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        }

        ~ScopedFieldRenderState() {
            glPopAttrib();
            glUseProgram(static_cast<GLuint>(m_program));
        }

    private:
        GLint m_program = 0;
    };

    void vertex(const glm::vec3& p) {
        glVertex3f(p.x, p.y, p.z);
    }

    float glyphFraction(double magnitude, double maximum,
        const VectorFieldRenderSettings& settings) {
        const double relative = (std::min)(1.0, magnitude / maximum);
        switch (settings.scale) {
        case VectorGlyphScale::RelativeMagnitude:
            return static_cast<float>(relative);
        case VectorGlyphScale::LogMagnitude: {
            const double strength = (std::max)(1.0, settings.logStrength);
            return static_cast<float>(std::log1p(strength * relative) /
                std::log1p(strength));
        }
        default:
            return 1.0f;
        }
    }
}

void FieldDebugRenderer::drawVector(const VectorField3D& field,
    const VectorFieldRenderSettings& settings) {
    if (!field.initialized() || settings.lengthInVoxels <= 0.0f) return;
    const double maximum = field.maxMagnitude();
    if (!(maximum > 0.0) || !std::isfinite(maximum)) return;

    ScopedFieldRenderState state;
    glLineWidth((std::max)(1.0f, settings.lineWidth));
    glColor4fv(&settings.color.x);
    glBegin(GL_LINES);
    for (unsigned int id = 0; id < field.size(); ++id) {
        const glm::vec3 value = field.get(id);
        const double magnitude = vectorMagnitude(value);
        if (!(magnitude > 0.0) || !std::isfinite(magnitude)) continue;

        SpatialVoxelRegion cell;
        if (!field.grid().region(id, cell)) continue;

        // Double arithmetic is necessary: SI B values from one elementary
        // charge can be ~1e-26 T; float length-squared would underflow.
        const glm::vec3 direction(
            static_cast<float>(static_cast<double>(value.x) / magnitude),
            static_cast<float>(static_cast<double>(value.y) / magnitude),
            static_cast<float>(static_cast<double>(value.z) / magnitude));
        const float length = settings.lengthInVoxels * field.grid().voxelEdgeM *
            glyphFraction(magnitude, maximum, settings);
        if (!(length > 0.0f)) continue;

        const glm::vec3 tip = cell.center + direction * length;
        const glm::vec3 reference = std::fabs(direction.y) < 0.9f ?
            glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
        const glm::vec3 side = glm::normalize(glm::cross(direction, reference));
        const glm::vec3 otherSide = glm::cross(direction, side);
        const glm::vec3 neck = tip - direction * (length * 0.25f);
        const float headWidth = length * 0.12f;

        vertex(cell.center); vertex(tip);
        vertex(tip); vertex(neck + side * headWidth);
        vertex(tip); vertex(neck - side * headWidth);
        vertex(tip); vertex(neck + otherSide * headWidth);
        vertex(tip); vertex(neck - otherSide * headWidth);
    }
    glEnd();
}

void FieldDebugRenderer::drawScalar(const ScalarField3D& field,
    const ScalarFieldRenderSettings& settings) {
    if (!field.initialized()) return;
    const double maximum = field.maxMagnitude();
    if (!(maximum > 0.0) || !std::isfinite(maximum)) return;

    ScopedFieldRenderState state;
    glEnable(GL_POINT_SMOOTH);
    const float minimumSize = (std::max)(1.0f, settings.minimumPointSize);
    const float maximumSize = (std::max)(minimumSize, settings.maximumPointSize);
    for (unsigned int id = 0; id < field.size(); ++id) {
        const double value = field.get(id);
        if (value == 0.0 || !std::isfinite(value)) continue;

        SpatialVoxelRegion cell;
        if (!field.grid().region(id, cell)) continue;
        const float relative = static_cast<float>((std::min)(1.0,
            std::fabs(value) / maximum));
        // Each selected field gets an independent relative display scale.
        // Charge sign stays visible even for SI magnitudes around 1e-18 C/m^3.
        const glm::vec4 color = value < 0.0 ?
            settings.negativeColor : settings.positiveColor;
        const float brightness = 0.35f + 0.65f * std::sqrt(relative);
        glColor4f(color.r * brightness, color.g * brightness,
            color.b * brightness, color.a);
        glPointSize(minimumSize + (maximumSize - minimumSize) * std::sqrt(relative));
        glBegin(GL_POINTS);
        vertex(cell.center);
        glEnd();
    }
}

void FieldDebugRenderer::drawProjectiles(EuclidRenderer& renderer,
    const std::vector<DebugProjectile>& projectiles) {
    std::vector<EuclidRenderer::DiagnosticParticleVisual> visuals;
    visuals.reserve(projectiles.size());
    for (const DebugProjectile& projectile : projectiles) {
        EuclidRenderer::DiagnosticParticleVisual visual;
        visual.position = projectile.position;
        visual.radius = projectile.renderRadiusM;
        visual.color = projectile.color;
        visual.emissiveIntensity = projectile.emissiveStrength;
        visuals.push_back(visual);
    }
    renderer.displayDiagnosticParticles(visuals);
}

void FieldDebugRenderer::drawCrosshair(int viewportWidth, int viewportHeight) {
    if (viewportWidth <= 0 || viewportHeight <= 0) return;
    ScopedFieldRenderState state;
    GLint matrixMode = GL_MODELVIEW;
    glGetIntegerv(GL_MATRIX_MODE, &matrixMode);
    glDisable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0.0, viewportWidth, viewportHeight, 0.0, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    const float x = static_cast<float>(viewportWidth) * 0.5f;
    const float y = static_cast<float>(viewportHeight) * 0.5f;
    glColor4f(0.9f, 1.0f, 1.0f, 0.9f);
    glLineWidth(1.5f);
    glBegin(GL_LINES);
    glVertex2f(x - 10.0f, y); glVertex2f(x - 3.0f, y);
    glVertex2f(x + 3.0f, y); glVertex2f(x + 10.0f, y);
    glVertex2f(x, y - 10.0f); glVertex2f(x, y - 3.0f);
    glVertex2f(x, y + 3.0f); glVertex2f(x, y + 10.0f);
    glEnd();

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(matrixMode);
}
