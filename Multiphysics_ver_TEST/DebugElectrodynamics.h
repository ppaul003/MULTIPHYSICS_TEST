#ifndef NDMSM_DEBUG_ELECTRODYNAMICS_H
#define NDMSM_DEBUG_ELECTRODYNAMICS_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "VoxelField3D.h"

enum class DebugProjectileSpecies { Electron, ArgonIon, ArgonNeutral };

struct DebugProjectile {
    DebugProjectileSpecies species = DebugProjectileSpecies::Electron;
    glm::vec3 position = glm::vec3(0.0f);  // m
    glm::vec3 velocity = glm::vec3(0.0f);  // m/s
    double chargeC = 0.0;
    double massKg = 0.0;
    // Visual marker properties, unrelated to atomic size or field softening.
    float renderRadiusM = 0.015f;
    glm::vec4 color = glm::vec4(1.0f);
    float emissiveStrength = 0.0f;
    std::uint64_t eventId = 0;
};

// Separate compact CPU diagnostics; normal CUDA marker ordering/capacity is
// unchanged. Every stored projectile is active; expired entries are erased.
class DebugElectrodynamics {
public:
    static constexpr std::size_t kCapacity = 128;
    enum class FireResult { Fired, MissedDomain, InvalidRay };

    FireResult fire(DebugProjectileSpecies species, const glm::vec3& originWorld,
        const glm::vec3& directionWorld, const SpatialVoxelGrid3D& grid);
    void update(double dt, const SpatialVoxelGrid3D& grid);
    void clear();
    const std::vector<DebugProjectile>& projectiles() const { return m_projectiles; }
    std::uint64_t firedCount() const { return m_firedCount; }
    float speedMps() const { return m_speedMps; }
    void setSpeedMps(float speed);

    // Nonrelativistic, softened quasi-static E/B diagnostics, not a Maxwell
    // solver. rho/J use NGP nearest-cell deposition. J=q*v is conventional
    // current, so for electrons it points OPPOSITE their velocity.
    void populateFields(const SpatialVoxelGrid3D& grid,
        VectorField3D& electricField, VectorField3D& magneticField,
        VectorField3D& currentDensity, ScalarField3D& chargeDensity) const;

private:
    std::vector<DebugProjectile> m_projectiles;
    std::uint64_t m_firedCount = 0;
    float m_speedMps = 0.75f;
};

#endif
