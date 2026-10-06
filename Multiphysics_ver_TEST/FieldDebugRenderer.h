#ifndef NDMSM_FIELD_DEBUG_RENDERER_H
#define NDMSM_FIELD_DEBUG_RENDERER_H

#include <glm/glm.hpp>
#include <vector>

class EuclidRenderer;
class ScalarField3D;
class VectorField3D;
struct DebugProjectile;

enum class VectorGlyphScale {
    Normalized,
    RelativeMagnitude,
    LogMagnitude
};

struct VectorFieldRenderSettings {
    VectorGlyphScale scale = VectorGlyphScale::Normalized;
    // Glyph dimensions are visual fractions of a voxel, never SI field values.
    float lengthInVoxels = 0.65f;
    glm::vec4 color = glm::vec4(0.2f, 0.9f, 1.0f, 0.85f);
    float lineWidth = 1.5f;
    double logStrength = 1000.0;
};

struct ScalarFieldRenderSettings {
    glm::vec4 positiveColor = glm::vec4(1.0f, 0.45f, 0.08f, 0.9f);
    glm::vec4 negativeColor = glm::vec4(0.15f, 0.55f, 1.0f, 0.9f);
    float minimumPointSize = 4.0f;
    float maximumPointSize = 13.0f;
};

class FieldDebugRenderer {
public:
    static void drawVector(const VectorField3D& field,
        const VectorFieldRenderSettings& settings = VectorFieldRenderSettings());
    static void drawScalar(const ScalarField3D& field,
        const ScalarFieldRenderSettings& settings = ScalarFieldRenderSettings());
    static void drawProjectiles(EuclidRenderer& renderer,
        const std::vector<DebugProjectile>& projectiles);
    static void drawCrosshair(int viewportWidth, int viewportHeight);
};

#endif
