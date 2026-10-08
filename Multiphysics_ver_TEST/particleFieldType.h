#ifndef _MPHY_PARTICLE_FIELD_H_
#define _MPHY_PARTICLE_FIELD_H_
#include <cstdint>

// Coupling metadata only; geometry and waves live in paramsEM_kernel.cuh.
enum class FieldBackend : uint8_t { CPU, CUDA };
enum class ElectrostaticMode : uint8_t { Off, DirectCoulombDebug, GridField };
enum class ElementId : uint16_t { Hydrogen = 1, Helium = 2, Argon = 18 };
enum class ParticleKind : uint8_t { Inactive, Atomic, Electron };

struct ParticleFieldMarker {
    float chargeC;
    float chargeToMass; // q/m [C/kg]; zero for neutral atoms
    uint32_t atomicId;
    uint16_t atomicNumber;
    int8_t chargeState;
    ParticleKind kind;
};
#endif
