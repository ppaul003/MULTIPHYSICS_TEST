#include "VoxelField3D.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {
    void validateGrid(const SpatialVoxelGrid3D& grid) {
        if (grid.voxelCount() == 0 || !std::isfinite(grid.voxelEdgeM) ||
            grid.voxelEdgeM <= 0.0f || !std::isfinite(grid.metersPerWorldUnit) || grid.metersPerWorldUnit <= 0.0) {
            throw std::invalid_argument("Voxel fields require a nonempty physical grid and positive spacing");
        }
    }

    unsigned int flatId(const SpatialVoxelGrid3D& grid, const glm::ivec3& index) {
        if (index.x < 0 || index.y < 0 || index.z < 0 ||
            index.x >= grid.dimensions.x || index.y >= grid.dimensions.y ||
            index.z >= grid.dimensions.z) {
            throw std::out_of_range("Physical voxel index is outside the field");
        }
        // Same X-fastest layout as SpatialVoxelGrid3D::region().
        return static_cast<unsigned int>(index.x + grid.dimensions.x *
            (index.y + grid.dimensions.y * index.z));
    }

    glm::vec3 derivative(const VectorField3D& field, glm::ivec3 index, int axis) {
        const SpatialVoxelGrid3D& grid = field.grid();
        if (grid.dimensions[axis] == 1) return glm::vec3(0.0f);
        glm::ivec3 low = index;
        glm::ivec3 high = index;
        low[axis] = (std::max)(0, index[axis] - 1);
        high[axis] = (std::min)(grid.dimensions[axis] - 1, index[axis] + 1);
        const float distanceM = static_cast<float>(high[axis] - low[axis]) * grid.voxelEdgeM * static_cast<float>(grid.metersPerWorldUnit);
        return (field.get(high) - field.get(low)) / distanceM;
    }
}

void ScalarField3D::initialize(const SpatialVoxelGrid3D& grid) {
    validateGrid(grid);
    m_values.assign(grid.voxelCount(), 0.0);
    m_grid = &grid;
}

const SpatialVoxelGrid3D& ScalarField3D::grid() const {
    if (!m_grid) throw std::logic_error("Scalar field is not initialized");
    return *m_grid;
}

double ScalarField3D::get(unsigned int id) const { return m_values.at(id); }
double ScalarField3D::get(const glm::ivec3& index) const { return get(flatId(grid(), index)); }
void ScalarField3D::set(unsigned int id, double value) { m_values.at(id) = value; }
void ScalarField3D::set(const glm::ivec3& index, double value) { set(flatId(grid(), index), value); }
void ScalarField3D::clear(double value) { std::fill(m_values.begin(), m_values.end(), value); }

double ScalarField3D::maxMagnitude() const {
    double result = 0.0;
    for (double value : m_values) result = (std::max)(result, std::abs(value));
    return result;
}

void VectorField3D::initialize(const SpatialVoxelGrid3D& grid) {
    validateGrid(grid);
    m_values.assign(grid.voxelCount(), glm::vec3(0.0f));
    m_grid = &grid;
}

const SpatialVoxelGrid3D& VectorField3D::grid() const {
    if (!m_grid) throw std::logic_error("Vector field is not initialized");
    return *m_grid;
}

const glm::vec3& VectorField3D::get(unsigned int id) const { return m_values.at(id); }
const glm::vec3& VectorField3D::get(const glm::ivec3& index) const { return get(flatId(grid(), index)); }
void VectorField3D::set(unsigned int id, const glm::vec3& value) { m_values.at(id) = value; }
void VectorField3D::set(const glm::ivec3& index, const glm::vec3& value) { set(flatId(grid(), index), value); }
void VectorField3D::clear(const glm::vec3& value) { std::fill(m_values.begin(), m_values.end(), value); }

double vectorMagnitude(const glm::vec3& value) {
    const double x = value.x, y = value.y, z = value.z;
    return std::sqrt(x * x + y * y + z * z);
}

double VectorField3D::maxMagnitude() const {
    double result = 0.0;
    for (const glm::vec3& value : m_values) {
        result = (std::max)(result, vectorMagnitude(value));
    }
    return result;
}

VectorField3D curl(const VectorField3D& field) {
    VectorField3D result(field.grid());
    computeCurl(field, result);
    return result;
}

void computeCurl(const VectorField3D& field, VectorField3D& result) {
    const SpatialVoxelGrid3D& grid = field.grid();
    validateGrid(grid);
    if (field.size() != grid.voxelCount()) {
        throw std::logic_error("Physical grid dimensions changed; reinitialize the field");
    }
    // Alias-safe reuse is useful when exploring operators interactively.
    if (&field == &result) {
        result = curl(field);
        return;
    }
    if (!result.initialized() || &result.grid() != &grid || result.size() != field.size()) {
        result.initialize(grid);
    }
    for (unsigned int id = 0; id < grid.voxelCount(); ++id) {
        SpatialVoxelRegion cell;
        grid.region(id, cell);
        const glm::vec3 dx = derivative(field, cell.index, 0);
        const glm::vec3 dy = derivative(field, cell.index, 1);
        const glm::vec3 dz = derivative(field, cell.index, 2);
        result.set(id, glm::vec3(dy.z - dz.y, dz.x - dx.z, dx.y - dy.x));
    }
}
