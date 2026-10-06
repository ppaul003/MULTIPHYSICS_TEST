#ifndef PARAMS_KERNEL_CUH
#define PARAMS_KERNEL_CUH

#include <cuda_runtime.h>
#include <vector_types.h>

typedef unsigned int uint;
typedef unsigned char uchar;


#define G 1.0f
#define EPS2 1e-4f
#define USE_TEX 0
#define SAMPLE_VOLUME 1

#ifndef USE_TEX
#define USE_TEX 0
#endif

#if USE_TEX
#define FETCH(t, i) tex1Dfetch(t##Tex, (i))
#else
#define FETCH(t, i) ((t)[(i)])
#endif

struct SimParams {
	// --- Uniform Grid Setup --- //
	uint numCells;
	uint numBodies;
	uint maxParticlesPerCell;

	uint3 gridSize;

	float3 gravity;
	float3 cellSize;
	float3 worldOrigin;

	// ---	EM CELL --- //
	float3 E; // Electric-field vector
	float3 B; // Magnetic-field vector

	float rho;
	float epsilon;
	float u;
	float sigma;

	float ne;
	float Te;

	// --- Legacy Particle sim parameters --- //
	float particleRadius;

	float globalDamping;

	float shear;
	float spring;
	float damping;
	float boundary;
	float attraction;
	float boundaryDamping;
};

#endif
