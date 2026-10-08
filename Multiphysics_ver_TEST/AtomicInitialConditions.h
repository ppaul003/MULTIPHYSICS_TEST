#ifndef ATOMIC_INITIAL_CONDITIONS_H
#define ATOMIC_INITIAL_CONDITIONS_H

#include <cmath>
#include <random>
#include <string>

// Pure initialization arithmetic; no field/particle ownership or runtime physics.
namespace AtomicInitialization {
enum class DensityType { Electron = 0, Neutral, Ion, Count };
constexpr double kMicrometerToMeter = 1.0e-6;
constexpr int kMinimumExponent = -308, kMaximumExponent = 308;
constexpr unsigned kHeavyParticleLimit = 21000;
constexpr unsigned kParticleCapacity = 49152;
constexpr unsigned kFullyIonizedMarkerCount = 2 * kHeavyParticleLimit;
constexpr unsigned kMarkerReserve = kParticleCapacity - kFullyIonizedMarkerCount;
constexpr unsigned kPopulationRoundingSeed = 1973;
static_assert(kMarkerReserve == 7152, "Preserve the reaction-slot reservoir");

inline const char* densityTypeName(DensityType type) {
    switch (type) {
    case DensityType::Electron: return "ELECTRON";
    case DensityType::Neutral: return "NEUTRAL";
    case DensityType::Ion: return "ION";
    default: return "INVALID";
    }
}

struct Population {
    DensityType densityType = DensityType::Electron;
    double enteredDensityM3 = 0.0, physicalVolumeM3 = 0.0;
    double expectedHeavyCount = 0.0, expectedNeutralCount = 0.0;
    double expectedIonCount = 0.0, expectedElectronCount = 0.0;
    unsigned heavyCount = 0, neutralCount = 0, ionCount = 0, electronCount = 0;
    unsigned activeMarkerCount = 0;
};

inline double physicalVolumeM3(double boxEdgeWorld) {
    // Same physical scale used by atomic CUDA/CPU fields and particle motion.
    // Render and collision coordinates remain in world units.
    const double edgeM = boxEdgeWorld * kMicrometerToMeter;
    return edgeM * edgeM * edgeM;
}

inline bool numberDensity(double mantissa, int exponent, double& density) {
    if (!std::isfinite(mantissa) || mantissa < 0.0 ||
        exponent < kMinimumExponent || exponent > kMaximumExponent) return false;
    density = mantissa * std::pow(10.0, exponent);
    return std::isfinite(density) && (mantissa == 0.0 || density > 0.0);
}

// Two fixed mt19937 draws form a portable 53-bit uniform [0,1).
// Quantization of the Bernoulli probability is below double precision.
inline double stochasticRound(double expected, std::mt19937& generator) {
    const double a = static_cast<double>(generator() >> 5);
    const double b = static_cast<double>(generator() >> 6);
    const double uniform = (a * 67108864.0 + b) / 9007199254740992.0;
    const double base = std::floor(expected);
    return base + (uniform < expected - base ? 1.0 : 0.0);
}

inline bool resolve(DensityType type, double mantissa, int exponent, double alpha,
    double boxEdgeWorld, Population& result, std::string& error) {
    result = {};
    error.clear();
    auto fail = [&](const char* reason) { error = reason; return false; };
    if (type < DensityType::Electron || type >= DensityType::Count)
        return fail("INVALID DENSITY TYPE");
    double density = 0.0;
    if (!numberDensity(mantissa, exponent, density))
        return fail("INVALID DENSITY (FINITE, >= 0; EXPONENT -308..308)");
    if (!std::isfinite(alpha) || alpha < 0.0 || alpha > 1.0)
        return fail("IONIZATION FRACTION MUST BE BETWEEN 0 AND 1");
    const double volume = physicalVolumeM3(boxEdgeWorld);
    if (!std::isfinite(boxEdgeWorld) || boxEdgeWorld <= 0.0 || !std::isfinite(volume) || volume <= 0.0)
        return fail("INVALID PHYSICAL BOX VOLUME");
    const double divisor = type == DensityType::Neutral ? 1.0 - alpha : alpha;
    if (divisor == 0.0 && density > 0.0)
        return fail(type == DensityType::Neutral
            ? "NONZERO NEUTRAL DENSITY REQUIRES ALPHA < 1"
            : "NONZERO ELECTRON/ION DENSITY REQUIRES ALPHA > 0");
    // Zero density at an underdetermined endpoint explicitly means empty gas.
    const double heavyDensity = divisor > 0.0 ? density / divisor : 0.0;
    const double ionDensity = type == DensityType::Neutral ? alpha * heavyDensity : density;
    const double neutralDensity = type == DensityType::Neutral ? density : heavyDensity - ionDensity;
    if (!std::isfinite(heavyDensity)) return fail("DERIVED DENSITY OVERFLOW");
    Population p;
    p.densityType = type;
    p.enteredDensityM3 = density;
    p.physicalVolumeM3 = volume;
    p.expectedHeavyCount = heavyDensity * volume;
    p.expectedNeutralCount = neutralDensity * volume;
    p.expectedIonCount = p.expectedElectronCount = ionDensity * volume;
    if (!std::isfinite(p.expectedHeavyCount) || !std::isfinite(p.expectedNeutralCount) ||
        !std::isfinite(p.expectedIonCount)) return fail("EXPECTED POPULATION OVERFLOW");
    std::mt19937 generator(kPopulationRoundingSeed);
    const double heavy = stochasticRound(p.expectedHeavyCount, generator);
    const double ions = stochasticRound(alpha * heavy, generator);
    // Compare in double before any integer conversion, including enormous inputs.
    if (heavy + ions > kParticleCapacity)
        return fail("REQUIRED MARKERS EXCEED PARTICLE CAPACITY");
    if (heavy > kHeavyParticleLimit)
        return fail("DENSITY EXCEEDS HEAVY PARTICLE LIMIT (21000)");
    p.heavyCount = static_cast<unsigned>(heavy);
    p.ionCount = p.electronCount = static_cast<unsigned>(ions);
    p.neutralCount = p.heavyCount - p.ionCount;
    p.activeMarkerCount = p.heavyCount + p.electronCount;
    result = p;
    return true;
}
} // namespace AtomicInitialization
#endif
