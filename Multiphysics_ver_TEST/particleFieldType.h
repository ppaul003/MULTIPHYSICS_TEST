#ifndef _MPHY_PARTICLE_FIELD_H_
#define _MPHY_PARTICLE_FIELD_H_

#include <cstdint>
#include <vector_types.h>

enum class FieldBackend : uint8_t {
	CPU,
	CUDA
};

enum class ElectrostaticMode : uint8_t {
	Off,
	DirectCoulombDebug,
	GridField
};

enum class ParticleKind : uint8_t {
	Hydrogen = 1,
	Helium = 2,
	Argon = 18
};

struct ParticleFieldGrid {
	uint3 dimensions;
	float3 origin;
	float cellSizeM;
	unsigned int cellCount;
};

struct ParticleFieldMarker {
	float chargeC;
	float chargeToMass; // q/m

	uint32_t atomicId;

	uint16_t atomicNumber;
	uint8_t chargeState;
	uint8_t kind;
};

struct AnalyticWave {
	bool enabled = false;

	float amplitudeVm = 0.0f;
	float frequencyHz = 0.0f;

	float3 direction;
	float3 polarization;

	float phaseRad = 0.0f;
};

struct ParticleFieldSettings {
	FieldBackend backend = FieldBackend::CPU;
	ElectrostaticMode electrostatics = ElectrostaticMode::Off;

	unsigned int poissonIterations = 64;
	float colombSofteningM = 0.0f;

	AnalyticWave wave;
};
#endif
