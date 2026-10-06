#ifndef NDMSM_WORKSPACE_CONTEXT_EM_H
#define NDMSM_WORKSPACE_CONTEXT_EM_H

#include <array>

#include <glm/glm.hpp>

struct SpatialVoxelRegion {
    unsigned int id = 0;
    glm::ivec3 index = glm::ivec3(0);
    glm::vec3 minimum = glm::vec3(0.0f);
    glm::vec3 maximum = glm::vec3(0.0f);
    glm::vec3 center = glm::vec3(0.0f);
    glm::vec3 halfExtent = glm::vec3(0.0f);
    float volumeM3 = 0.0f;
};

// Physical workspace partition shared by particle and future multiphysics
// cartridges. IDs are X-fastest, followed by Y and then Z.
struct SpatialVoxelGrid3D {
    glm::ivec3 dimensions = glm::ivec3(8, 8, 8);
    glm::vec3 origin = glm::vec3(-2.0f, -2.0f, -2.0f);
    float voxelEdgeM = 0.5f;

    unsigned int voxelCount() const {
        if (dimensions.x <= 0 || dimensions.y <= 0 || dimensions.z <= 0) {
            return 0;
        }

        return static_cast<unsigned int>(
            dimensions.x * dimensions.y * dimensions.z
        );
    }

    bool region(unsigned int id, SpatialVoxelRegion& result) const {
        const unsigned int count = voxelCount();
        if (id >= count || voxelEdgeM <= 0.0f) {
            return false;
        }

        const unsigned int dimensionX =
            static_cast<unsigned int>(dimensions.x);
        const unsigned int dimensionY =
            static_cast<unsigned int>(dimensions.y);

        const unsigned int x = id % dimensionX;
        const unsigned int y = (id / dimensionX) % dimensionY;
        const unsigned int z = id / (dimensionX * dimensionY);

        result.id = id;
        result.index = glm::ivec3(
            static_cast<int>(x),
            static_cast<int>(y),
            static_cast<int>(z)
        );
        result.minimum = origin + glm::vec3(
            static_cast<float>(x),
            static_cast<float>(y),
            static_cast<float>(z)
        ) * voxelEdgeM;
        result.maximum = result.minimum + glm::vec3(voxelEdgeM);
        result.center = (result.minimum + result.maximum) * 0.5f;
        result.halfExtent = glm::vec3(voxelEdgeM * 0.5f);
        result.volumeM3 = voxelEdgeM * voxelEdgeM * voxelEdgeM;
        return true;
    }
};

// A user-selectable physical spawn volume assembled from eight base voxels.
// Constituents are ordered X-fastest, then Y, then Z within the local 2x2x2
// subdivision.
struct SpawnDensityRegion3D {
    unsigned int id = 0;
    glm::ivec3 index = glm::ivec3(0);
    glm::vec3 minimum = glm::vec3(0.0f);
    glm::vec3 maximum = glm::vec3(0.0f);
    glm::vec3 center = glm::vec3(0.0f);
    glm::vec3 halfExtent = glm::vec3(0.0f);
    float volumeM3 = 0.0f;
    std::array<SpatialVoxelRegion, 8> constituentBaseVoxels{};


};

// Composite selection grid layered over SpatialVoxelGrid3D. The base grid
// remains the reusable 8x8x8 physical partition; this view groups 2x2x2 base
// voxels into one continuous PARTICLE_SIM spawn-density region.
struct SpawnDensityRegionGrid3D {
    glm::ivec3 baseVoxelSpan = glm::ivec3(2, 2, 2);

    glm::ivec3 dimensions(const SpatialVoxelGrid3D& baseGrid) const {
        if (baseVoxelSpan.x <= 0 ||
            baseVoxelSpan.y <= 0 ||
            baseVoxelSpan.z <= 0 ||
            baseGrid.dimensions.x <= 0 ||
            baseGrid.dimensions.y <= 0 ||
            baseGrid.dimensions.z <= 0 ||
            baseGrid.dimensions.x % baseVoxelSpan.x != 0 ||
            baseGrid.dimensions.y % baseVoxelSpan.y != 0 ||
            baseGrid.dimensions.z % baseVoxelSpan.z != 0) {
            return glm::ivec3(0);
        }

        return glm::ivec3(
            baseGrid.dimensions.x / baseVoxelSpan.x,
            baseGrid.dimensions.y / baseVoxelSpan.y,
            baseGrid.dimensions.z / baseVoxelSpan.z
        );
    }

    unsigned int regionCount(const SpatialVoxelGrid3D& baseGrid) const {
        const glm::ivec3 regionDimensions = dimensions(baseGrid);
        return static_cast<unsigned int>(
            regionDimensions.x * 
            regionDimensions.y * 
            regionDimensions.z
        );
    }

    unsigned int selectionCount(const SpatialVoxelGrid3D& baseGrid) const {
        return regionCount(baseGrid) + 1;
    }

    bool region(const SpatialVoxelGrid3D& baseGrid,
        unsigned int id, SpawnDensityRegion3D& result) const {

        const glm::ivec3 regionDimensions = dimensions(baseGrid);
        const unsigned int count = regionCount(baseGrid);
        if (id >= count || baseGrid.voxelEdgeM <= 0.0f) {
            return false;
        }

        const unsigned int dimensionX = static_cast<unsigned int>(regionDimensions.x);
        const unsigned int dimensionY = static_cast<unsigned int>(regionDimensions.y);

        const unsigned int regionX = id % dimensionX;
        const unsigned int regionY = (id / dimensionX) % dimensionY;
        const unsigned int regionZ = id / (dimensionX * dimensionY);

        const glm::ivec3 baseOrigin(
            static_cast<int>(regionX) * baseVoxelSpan.x,
            static_cast<int>(regionY) * baseVoxelSpan.y,
            static_cast<int>(regionZ) * baseVoxelSpan.z
        );

        const std::array<glm::ivec3, 8> baseIndices = {{
            baseOrigin + glm::ivec3(0, 0, 0),
            baseOrigin + glm::ivec3(1, 0, 0),
            baseOrigin + glm::ivec3(0, 1, 0),
            baseOrigin + glm::ivec3(1, 1, 0),
            baseOrigin + glm::ivec3(0, 0, 1),
            baseOrigin + glm::ivec3(1, 0, 1),
            baseOrigin + glm::ivec3(0, 1, 1),
            baseOrigin + glm::ivec3(1, 1, 1)
        }};

        const unsigned int baseDimensionX =
            static_cast<unsigned int>(baseGrid.dimensions.x);
        const unsigned int baseDimensionY =
            static_cast<unsigned int>(baseGrid.dimensions.y);

        for (std::size_t i = 0; i < baseIndices.size(); i++) {
            const glm::ivec3& baseIndex = baseIndices[i];
            const unsigned int baseId =
                static_cast<unsigned int>(baseIndex.x) +
                baseDimensionX * static_cast<unsigned int>(baseIndex.y) +
                baseDimensionX * baseDimensionY *
                    static_cast<unsigned int>(baseIndex.z);

            if (!baseGrid.region(baseId, result.constituentBaseVoxels[i])) {
                return false;
            }
        }

        const glm::vec3 regionSize =
            glm::vec3(baseVoxelSpan) * baseGrid.voxelEdgeM;

        result.id = id;

        result.index = glm::ivec3(
            static_cast<int>(regionX),
            static_cast<int>(regionY),
            static_cast<int>(regionZ)
        );

        result.minimum = baseGrid.origin + glm::vec3(baseOrigin) * baseGrid.voxelEdgeM;
        result.maximum = result.minimum + regionSize;
        result.center = (result.minimum + result.maximum) * 0.5f;
        result.halfExtent = regionSize * 0.5f;
        result.volumeM3 = regionSize.x * regionSize.y * regionSize.z;
        return true;
    }

    bool centeredRegion(
        const SpatialVoxelGrid3D& baseGrid, 
        SpawnDensityRegion3D& result) const {

        if (baseVoxelSpan.x <= 0 ||
            baseVoxelSpan.y <= 0 ||
            baseVoxelSpan.z <= 0 ||
            baseGrid.dimensions.x < baseVoxelSpan.x ||
            baseGrid.dimensions.y < baseVoxelSpan.y ||
            baseGrid.dimensions.z < baseVoxelSpan.z ||
            baseGrid.voxelEdgeM <= 0.0f) {
            return false;
        }

        // Difference must be even along every dimension
        // so the selected block can be centered exactly.
        const glm::ivec3 remaining =
            baseGrid.dimensions - baseVoxelSpan;

        if ((remaining.x % 2) != 0 ||
            (remaining.y % 2) != 0 ||
            (remaining.z % 2) != 0) {
            return false;
        }

        // 8x8x8 base grid with a 2x2x2 spawn volume:
        //
        // (8 - 2) / 2 = 3
        //
        // Therefore the centered region starts at
        // base voxel index (3,3,3).
        const glm::ivec3 baseOrigin =
            remaining / 2;

        const std::array<glm::ivec3, 8> baseIndices = { {
            baseOrigin + glm::ivec3(0, 0, 0),
            baseOrigin + glm::ivec3(1, 0, 0),
            baseOrigin + glm::ivec3(0, 1, 0),
            baseOrigin + glm::ivec3(1, 1, 0),

            baseOrigin + glm::ivec3(0, 0, 1),
            baseOrigin + glm::ivec3(1, 0, 1),
            baseOrigin + glm::ivec3(0, 1, 1),
            baseOrigin + glm::ivec3(1, 1, 1)
        } };

        const unsigned int baseDimensionX = static_cast<unsigned int>(baseGrid.dimensions.x);
        const unsigned int baseDimensionY = static_cast<unsigned int>(baseGrid.dimensions.y);

        for (std::size_t i = 0; i < baseIndices.size(); i++) {
            const glm::ivec3& baseIndex = baseIndices[i];
            const unsigned int baseId = static_cast<unsigned int>(baseIndex.x) +
                baseDimensionX * static_cast<unsigned int>(baseIndex.y) +
                baseDimensionX * baseDimensionY * static_cast<unsigned int>(baseIndex.z);

            if (!baseGrid.region(baseId, result.constituentBaseVoxels[i])) {
                return false;
            }
        }

        const glm::vec3 regionSize = glm::vec3(baseVoxelSpan) * baseGrid.voxelEdgeM;
        result.id = regionCount(baseGrid);

        // Special region: not a regular 4x4x4 region index.
        result.index = glm::ivec3(-1);
        result.minimum = baseGrid.origin +
            glm::vec3(baseOrigin) * baseGrid.voxelEdgeM;

        result.maximum = result.minimum + regionSize;
        result.center = (result.minimum + result.maximum) *  0.5f;
        result.halfExtent = regionSize * 0.5f;

        result.volumeM3 =
            regionSize.x *
            regionSize.y *
            regionSize.z;

        return true;
    }

    bool selection(
        const SpatialVoxelGrid3D& baseGrid, 
        unsigned int selectionIndex, 
        SpawnDensityRegion3D& result) const {

        // Selection 0 is the special center volume.
        if (selectionIndex == 0) {
            return centeredRegion(baseGrid, result);
        }

        // Everything after CENTER maps to the existing
        // physical region IDs.
        //
        // selection 1  -> region 0
        // selection 2  -> region 1
        // ...
        // selection 64 -> region 63
        const unsigned int regularRegionId = selectionIndex - 1;

        return region(baseGrid, regularRegionId, result);
    }
};

class EuclidRenderer;
class TheArbiter;
class ViewPort;
class CameraProcessor;

struct WorkspaceFrameContext {
    float deltaTime = 0.0f;
    float elapsedTime = 0.0f;
    int viewportWidth = 1920;
    int viewportHeight = 1080;
    bool displayEnabled = true;
};

struct WorkspaceServices {
    EuclidRenderer* renderer = nullptr;
    TheArbiter* arbiter = nullptr;
    ViewPort* viewport = nullptr;
    CameraProcessor* camera = nullptr;
};

#endif
