#ifndef NDMSM_FIELD_DEBUG_RENDERER_H
#define NDMSM_FIELD_DEBUG_RENDERER_H

#include <glm/glm.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

class EuclidRenderer;
class ScalarField3D;
class VectorField3D;
class FieldSystem;
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
    // Same SI units as the selected field. A positive finite reference keeps
    // magnitude scaling stable as sources move; zero uses the frame maximum.
    double referenceMagnitude = 0.0;
};

// Pure display scaling, shared with CPU sanity checks. Physical field and
// velocity values are never modified to achieve a visible arrow length.
namespace FieldGlyphDisplay {
    inline float vectorLengthFraction(double magnitude, double frameMaximum,
        const VectorFieldRenderSettings& settings) {
        if (!(magnitude > 0.0) || !std::isfinite(magnitude)) return 0.0f;
        if (settings.scale == VectorGlyphScale::Normalized) return 1.0f;
        const double reference = settings.referenceMagnitude > 0.0 &&
            std::isfinite(settings.referenceMagnitude) ?
            settings.referenceMagnitude : frameMaximum;
        if (!(reference > 0.0) || !std::isfinite(reference)) return 0.0f;
        const double relative = (std::min)(1.0, magnitude / reference);
        if (settings.scale == VectorGlyphScale::LogMagnitude) {
            const double strength = std::isfinite(settings.logStrength) ?
                (std::max)(1.0, settings.logStrength) : 1000.0;
            return static_cast<float>(std::log1p(strength * relative) /
                std::log1p(strength));
        }
        return static_cast<float>(relative);
    }

    constexpr double kVelocityReferenceSpeedMps = 1.0;
    constexpr float kMaximumVelocityLengthInVoxels = 1.25f;

    inline float velocityLengthInVoxels(double speedMps) {
        if (!(speedMps > 0.0) || !std::isfinite(speedMps)) return 0.0f;
        return kMaximumVelocityLengthInVoxels * static_cast<float>(
            speedMps / (speedMps + kVelocityReferenceSpeedMps));
    }
}

struct ScalarFieldRenderSettings {
    glm::vec4 positiveColor = glm::vec4(1.0f, 0.45f, 0.08f, 0.9f);
    glm::vec4 negativeColor = glm::vec4(0.15f, 0.55f, 1.0f, 0.9f);
    float minimumPointSize = 4.0f;
    float maximumPointSize = 13.0f;
};

class FieldDebugRenderer {
public:
    // FieldSystem owns CUDA/GL lifetime and builds the buffers before drawing.
    static void drawElectricField(const FieldSystem& fields);
    static void drawScalarField(const FieldSystem& fields, float pointSize = 5.0f);
    static void drawVector(const VectorField3D& field,
        const VectorFieldRenderSettings& settings = VectorFieldRenderSettings());
    // A mismatched color count safely falls back to the uniform color.
    static void drawVector(const VectorField3D& field,
        const std::vector<glm::vec4>& perVoxelColors,
        const VectorFieldRenderSettings& settings = VectorFieldRenderSettings());
    static void drawScalar(const ScalarField3D& field,
        const ScalarFieldRenderSettings& settings = ScalarFieldRenderSettings());
    static void drawProjectiles(EuclidRenderer& renderer,
        const std::vector<DebugProjectile>& projectiles);
    static void drawProjectileVelocities(const std::vector<DebugProjectile>& projectiles,
        float voxelEdgeM);
    static void drawCrosshair(int viewportWidth, int viewportHeight);
};

#endif
