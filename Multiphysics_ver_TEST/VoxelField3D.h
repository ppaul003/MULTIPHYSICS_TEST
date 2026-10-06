#ifndef NDMSM_VOXEL_FIELD_3D_H
#define NDMSM_VOXEL_FIELD_3D_H

#include <cstddef>
#include <vector>

#include "WorkspaceContextEM.h"

// Fields borrow the physical grid; it must outlive them. No geometry is copied.
// Reinitialize after changing its dimensions. SI values are never display-scaled.
class ScalarField3D {
public:
    ScalarField3D() = default;
    explicit ScalarField3D(const SpatialVoxelGrid3D& grid) { initialize(grid); }

    void initialize(const SpatialVoxelGrid3D& grid);
    bool initialized() const { return m_grid != nullptr; }
    const SpatialVoxelGrid3D& grid() const;
    std::size_t size() const { return m_values.size(); }
    double get(unsigned int id) const;
    double get(const glm::ivec3& index) const;
    void set(unsigned int id, double value);
    void set(const glm::ivec3& index, double value);
    void clear(double value = 0.0);
    double maxMagnitude() const;

private:
    const SpatialVoxelGrid3D* m_grid = nullptr;
    std::vector<double> m_values;
};

class VectorField3D {
public:
    VectorField3D() = default;
    explicit VectorField3D(const SpatialVoxelGrid3D& grid) { initialize(grid); }

    void initialize(const SpatialVoxelGrid3D& grid);
    bool initialized() const { return m_grid != nullptr; }
    const SpatialVoxelGrid3D& grid() const;
    std::size_t size() const { return m_values.size(); }
    const glm::vec3& get(unsigned int id) const;
    const glm::vec3& get(const glm::ivec3& index) const;
    void set(unsigned int id, const glm::vec3& value);
    void set(const glm::ivec3& index, const glm::vec3& value);
    void clear(const glm::vec3& value = glm::vec3(0.0f));
    double maxMagnitude() const;

private:
    const SpatialVoxelGrid3D* m_grid = nullptr;
    std::vector<glm::vec3> m_values;
};

// Double intermediates avoid underflow when squaring single-particle SI B/J.
double vectorMagnitude(const glm::vec3& value);

// Interior: centered differences. Outer faces: first-order one-sided
// differences. A one-cell axis has zero derivative; faces never wrap.
VectorField3D curl(const VectorField3D& field);
void computeCurl(const VectorField3D& field, VectorField3D& result);

#endif
