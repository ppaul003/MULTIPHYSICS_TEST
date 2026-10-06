#include "DebugElectrodynamics.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {
    constexpr double kElementaryChargeC = 1.602176634e-19;
    constexpr double kElectronMassKg = 9.1093837e-31;
    constexpr double kArgonMassKg = 39.948 * 1.66053906660e-27;
    constexpr double kCoulombFactor = 8.9875517923e9;
    constexpr double kMu0OverFourPi = 1.0e-7;

    bool finiteVector(const glm::vec3& value) {
        return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
    }

    bool domainBounds(const SpatialVoxelGrid3D& grid, glm::vec3& minimum, glm::vec3& maximum) {
        if (grid.voxelCount() == 0 || !std::isfinite(grid.voxelEdgeM) ||
            !finiteVector(grid.origin)) return false;
        SpatialVoxelRegion first, last;
        if (!grid.region(0, first) || !grid.region(grid.voxelCount() - 1, last)) return false;
        minimum = first.minimum;
        maximum = last.maximum;
        return finiteVector(maximum);
    }

    // Half-open domain agrees with containing-cell indexing at external faces.
    bool inside(const glm::vec3& position, const glm::vec3& minimum, const glm::vec3& maximum) {
        return finiteVector(position) && position.x >= minimum.x && position.x < maximum.x &&
            position.y >= minimum.y && position.y < maximum.y &&
            position.z >= minimum.z && position.z < maximum.z;
    }

    template<class Field> void prepareField(Field& field, const SpatialVoxelGrid3D& grid) {
        if (!field.initialized() || &field.grid() != &grid || field.size() != grid.voxelCount()) {
            field.initialize(grid);
        }
        else field.clear();
    }

    DebugProjectile speciesProperties(DebugProjectileSpecies species) {
        DebugProjectile result;
        result.species = species;
        if (species == DebugProjectileSpecies::Electron) {
            result.chargeC = -kElementaryChargeC;
            result.massKg = kElectronMassKg;
            result.renderRadiusM = 0.015f;
            result.color = glm::vec4(0.05f, 0.25f, 1.0f, 1.0f);
            result.emissiveStrength = 2.5f;
        }
        else {
            result.massKg = kArgonMassKg;
            result.renderRadiusM = 0.03f;
            if (species == DebugProjectileSpecies::ArgonIon) {
                result.chargeC = kElementaryChargeC;
                result.color = glm::vec4(0.05f, 1.0f, 0.02f, 1.0f);
                result.emissiveStrength = 2.5f;
            }
            else {
                result.color = glm::vec4(1.0f, 0.05f, 0.0f, 1.0f);
            }
        }
        return result;
    }
}

DebugElectrodynamics::FireResult DebugElectrodynamics::fire(
    DebugProjectileSpecies species, const glm::vec3& originWorld,
    const glm::vec3& directionWorld, const SpatialVoxelGrid3D& grid) {
    const double magnitude = vectorMagnitude(directionWorld);
    glm::vec3 minimum, maximum;
    if (!finiteVector(originWorld) || !finiteVector(directionWorld) ||
        magnitude <= 0.0 || !domainBounds(grid, minimum, maximum)) {
        return FireResult::InvalidRay;
    }
    const glm::dvec3 origin(originWorld);
    const glm::dvec3 direction = glm::dvec3(directionWorld) / magnitude;
    double entry = 0.0;
    double exit = (std::numeric_limits<double>::infinity)();
    for (int axis = 0; axis < 3; ++axis) {
        if (std::abs(direction[axis]) < 1.0e-12) {
            if (origin[axis] < minimum[axis] || origin[axis] > maximum[axis]) {
                return FireResult::MissedDomain;
            }
            continue;
        }
        double nearT = (minimum[axis] - origin[axis]) / direction[axis];
        double farT = (maximum[axis] - origin[axis]) / direction[axis];
        if (nearT > farT) std::swap(nearT, farT);
        entry = (std::max)(entry, nearT);
        exit = (std::min)(exit, farT);
        if (exit <= entry) return FireResult::MissedDomain;
    }
    const double offsetM = (std::min)(static_cast<double>(grid.voxelEdgeM) * 1.0e-4,
        (exit - entry) * 0.5);
    const glm::vec3 spawn(origin + direction * (entry + offsetM));
    if (!inside(spawn, minimum, maximum)) return FireResult::MissedDomain;
    DebugProjectile projectile = speciesProperties(species);
    projectile.position = spawn;
    projectile.velocity = glm::vec3(direction * static_cast<double>(m_speedMps));
    projectile.eventId = ++m_firedCount;
    if (m_projectiles.size() == kCapacity) m_projectiles.erase(m_projectiles.begin());
    m_projectiles.push_back(projectile);
    return FireResult::Fired;
}

void DebugElectrodynamics::setSpeedMps(float speed) {
    if (!std::isfinite(speed) || speed < 0.0f) {
        throw std::invalid_argument("Diagnostic projectile speed must be finite and nonnegative");
    }
    m_speedMps = speed;
}

void DebugElectrodynamics::update(double dt, const SpatialVoxelGrid3D& grid) {
    if (!std::isfinite(dt) || dt <= 0.0) return;
    glm::vec3 minimum, maximum;
    if (!domainBounds(grid, minimum, maximum)) return;
    for (DebugProjectile& projectile : m_projectiles) {
        projectile.position = glm::vec3(glm::dvec3(projectile.position) + glm::dvec3(projectile.velocity) * dt);
    }
    m_projectiles.erase(std::remove_if(m_projectiles.begin(), m_projectiles.end(),
        [&](const DebugProjectile& projectile) { return !inside(projectile.position, minimum, maximum); }),
        m_projectiles.end());
}

void DebugElectrodynamics::clear() {
    m_projectiles.clear();
    m_firedCount = 0;
}

void DebugElectrodynamics::populateFields(const SpatialVoxelGrid3D& grid,
    VectorField3D& electricField, VectorField3D& magneticField,
    VectorField3D& currentDensity, ScalarField3D& chargeDensity) const {
    prepareField(electricField, grid);
    prepareField(magneticField, grid);
    prepareField(currentDensity, grid);
    prepareField(chargeDensity, grid);

    // One quarter of a physical voxel edge is numerical debug softening,
    // never an atomic radius or a modification of the stored particle charge.
    const double epsilonM = 0.25 * static_cast<double>(grid.voxelEdgeM);
    for (unsigned int id = 0; id < grid.voxelCount(); ++id) {
        SpatialVoxelRegion cell;
        grid.region(id, cell);
        glm::dvec3 electric(0.0), magnetic(0.0), current(0.0);
        double charge = 0.0;
        for (const DebugProjectile& projectile : m_projectiles) {
            if (projectile.chargeC == 0.0) continue;
            const glm::dvec3 displacement = glm::dvec3(cell.center) - glm::dvec3(projectile.position);
            const double r2 = glm::dot(displacement, displacement) + epsilonM * epsilonM;
            const double denominator = r2 * std::sqrt(r2);
            electric += kCoulombFactor * projectile.chargeC * displacement / denominator;
            magnetic += kMu0OverFourPi * projectile.chargeC *
                glm::cross(glm::dvec3(projectile.velocity), displacement) / denominator;
            // NGP / nearest-cell debug deposition; sum all particles in this
            // half-open cell. This is neither CIC nor a self-consistent PIC step.
            if (inside(projectile.position, cell.minimum, cell.maximum)) {
                charge += projectile.chargeC / static_cast<double>(cell.volumeM3);
                current += projectile.chargeC * glm::dvec3(projectile.velocity) /
                    static_cast<double>(cell.volumeM3);
            }
        }
        electricField.set(id, glm::vec3(electric));
        magneticField.set(id, glm::vec3(magnetic));
        currentDensity.set(id, glm::vec3(current));
        chargeDensity.set(id, charge);
    }
}
