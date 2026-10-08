#include <GL/glew.h>

#include "FieldDebugRenderer.h"
#include "VoxelField3D.h"
#include "DebugElectrodynamics.h"
#include "rendererEM_Euclid.h"
#include "fieldSystem.h"
#include <cstddef>

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

    glm::vec3 unitDirection(const glm::vec3& value, double magnitude) {
        // SI B values from one elementary charge can be ~1e-26 T; float
        // length-squared would underflow, so normalize in double precision.
        return glm::vec3(
            static_cast<float>(static_cast<double>(value.x) / magnitude),
            static_cast<float>(static_cast<double>(value.y) / magnitude),
            static_cast<float>(static_cast<double>(value.z) / magnitude));
    }

    // Caller owns GL_LINES and render state; both field and velocity glyphs
    // use the same arrow shape, with an origin and a display-scaled length.
    void drawArrow(const glm::vec3& origin, const glm::vec3& direction, float length) {
        const glm::vec3 tip = origin + direction * length;
        const glm::vec3 reference = std::fabs(direction.y) < 0.9f ?
            glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
        const glm::vec3 side = glm::normalize(glm::cross(direction, reference));
        const glm::vec3 otherSide = glm::cross(direction, side);
        const glm::vec3 neck = tip - direction * (length * 0.25f);
        const float headWidth = length * 0.12f;

        vertex(origin); vertex(tip);
        vertex(tip); vertex(neck + side * headWidth);
        vertex(tip); vertex(neck - side * headWidth);
        vertex(tip); vertex(neck + otherSide * headWidth);
        vertex(tip); vertex(neck - otherSide * headWidth);
    }

    void drawVectorGlyphs(const VectorField3D& field,
        const VectorFieldRenderSettings& settings,
        const std::vector<glm::vec4>* perVoxelColors) {
        if (!field.initialized() || !(settings.lengthInVoxels > 0.0f) ||
            !std::isfinite(settings.lengthInVoxels)) return;
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
            const float length = settings.lengthInVoxels * field.grid().voxelEdgeM *
                FieldGlyphDisplay::vectorLengthFraction(magnitude, maximum, settings);
            if (!(length > 0.0f) || !std::isfinite(length)) continue;

            if (perVoxelColors) glColor4fv(&(*perVoxelColors)[id].x);
            drawArrow(cell.center, unitDirection(value, magnitude), length);
        }
        glEnd();
    }
}

void FieldDebugRenderer::drawVector(const VectorField3D& field,
    const VectorFieldRenderSettings& settings) {
    drawVectorGlyphs(field, settings, nullptr);
}

void FieldDebugRenderer::drawVector(const VectorField3D& field,
    const std::vector<glm::vec4>& perVoxelColors,
    const VectorFieldRenderSettings& settings) {
    drawVectorGlyphs(field, settings,
        perVoxelColors.size() == field.size() ? &perVoxelColors : nullptr);
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

void FieldDebugRenderer::drawProjectileVelocities(
    const std::vector<DebugProjectile>& projectiles, float voxelEdgeM) {
    if (!(voxelEdgeM > 0.0f) || !std::isfinite(voxelEdgeM) || projectiles.empty()) return;
    ScopedFieldRenderState state;
    glLineWidth(2.0f);
    glColor4f(1.0f, 0.45f, 0.08f, 0.95f);
    glBegin(GL_LINES);
    // Every projectile in the diagnostic container is active; update erases
    // expired sources. These const arrows cannot alter its SI velocity.
    for (const DebugProjectile& projectile : projectiles) {
        const double speed = vectorMagnitude(projectile.velocity);
        const float length = voxelEdgeM * FieldGlyphDisplay::velocityLengthInVoxels(speed);
        if (!(length > 0.0f) || !std::isfinite(length)) continue;
        drawArrow(projectile.position, unitDirection(projectile.velocity, speed), length);
    }
    glEnd();
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

// Consume already-built VBOs; no ownership or GPU-to-CPU field readback here.
namespace {
    void drawFieldVbo(unsigned vbo, unsigned count, GLenum primitive, float pointSize) {
        if (!vbo || !count) return;
        ScopedFieldRenderState state;
        GLint previousBuffer=0;
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING,&previousBuffer);
        glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT);
        glBindBuffer(GL_ARRAY_BUFFER,vbo);
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);
        glDisableClientState(GL_NORMAL_ARRAY);
        glVertexPointer(4,GL_FLOAT,sizeof(FieldGlyphVertex),
            reinterpret_cast<const void*>(offsetof(FieldGlyphVertex,position)));
        glColorPointer(4,GL_FLOAT,sizeof(FieldGlyphVertex),
            reinterpret_cast<const void*>(offsetof(FieldGlyphVertex,color)));
        glEnable(GL_ALPHA_TEST);
        glAlphaFunc(GL_GREATER,0.0f);
        glLineWidth(1.5f);
        glPointSize((std::max)(1.0f,pointSize));
        glDrawArrays(primitive,0,static_cast<GLsizei>(count));
        glPopClientAttrib();
        glBindBuffer(GL_ARRAY_BUFFER,previousBuffer);
    }
}
void FieldDebugRenderer::drawElectricField(const FieldSystem& fields) {
    drawFieldVbo(fields.getGlyphBuffer(),fields.getGlyphVertexCount(),GL_LINES,1);
}
void FieldDebugRenderer::drawScalarField(const FieldSystem& fields, float pointSize) {
    drawFieldVbo(fields.getScalarBuffer(),fields.getScalarVertexCount(),GL_POINTS,pointSize);
}

