#ifndef PARAMS_KERNEL_CUH
#define PARAMS_KERNEL_CUH

#include <cuda_runtime.h>
#include <vector_types.h>
#include <cstdint>
#include <type_traits>

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

//
// ============================================================
// PARTICLE SYSTEM
// ============================================================
//

// CUDA constants used by ParticleSystem kernels.
// These describe the particle collision/integration domain,
// NOT the EM/field grid.
//
struct ParticleSimParams {
    // --- Particle population ---
    uint32_t numBodies;

    // --- Collision / neighbor grid ---
    uint3 gridSize;
    uint32_t numCells;
    uint32_t maxParticlesPerCell;

    float3 worldOrigin;
    float3 cellSize;

    // --- External particle acceleration ---
    float3 gravity;

    // --- Particle collision model ---
    float particleRadius;

    float globalDamping;

    float shear;
    float spring;
    float damping;
    float attraction;

    // --- Domain boundary ---
    float boundary;
    float boundaryDamping;
};

//
// ============================================================
// FIELD SYSTEM GRID
// ============================================================
//

// Independent Eulerian field grid.
//
// IMPORTANT:
// This does NOT need to have the same resolution as the
// ParticleSystem collision grid.
//
struct FieldGridParams {
    uint3 dimensions;

    float3 origin;
    float3 cellSize;

    uint32_t cellCount;
};


//
// ============================================================
// FIELD SOLVER
// ============================================================
//

enum class FieldBoundaryMode : uint8_t {
    GroundedDirichlet,
    Neumann,
    Periodic
};

struct FieldSolverParams {
    FieldBoundaryMode boundaryMode;

    uint32_t poissonIterations;

    float coulombSofteningM;

    // Numerical/render-independent field threshold.
    // Only add if actually required by solver logic.
    float epsilon;
};




//
// ============================================================
// MATERIAL / MEDIUM PROPERTIES
// ============================================================
//

// These describe the medium occupying the field domain.
//
// If these later vary spatially, move them into field buffers.
//
struct MaterialFieldParams {
    float relativePermittivity = 1.0f;   // epsilon_r
    float relativePermeability = 1.0f;   // mu_r
    float conductivitySm = 0.0f;   // sigma [S/m]
};


//
// ============================================================
// EXTERNAL ELECTROMAGNETIC FIELD
// ============================================================
//

// Only use these for UNIFORM externally imposed fields.
//
// Spatially varying E/B belong in FieldBuffers.
//
struct UniformEMField {
    bool enabled = false;

    float3 electricVm = make_float3(0.0f, 0.0f, 0.0f);
    float3 magneticT = make_float3(0.0f, 0.0f, 0.0f);
};


//
// ============================================================
// ANALYTIC EM WAVE
// ============================================================
//

struct AnalyticWaveParams {
    bool enabled = false;

    float amplitudeVm = 0.0f;
    float frequencyHz = 0.0f;
    float phaseRad = 0.0f;

    float3 propagationDirection =
        make_float3(0.0f, 0.0f, 1.0f);

    float3 polarization =
        make_float3(1.0f, 0.0f, 0.0f);
};


//
// ============================================================
// FIELD SYSTEM GPU STORAGE
// ============================================================
//

// GPU-owned arrays.
//
// One element corresponds to one FieldGridParams cell.
//
struct FieldBuffers {
    // --- Scalar fields ---
    float* chargeDensity = nullptr;   // rho_q [C/m^3]
    float* electricPotential = nullptr; // phi [V]

    // Jacobi scratch buffer.
    float* electricPotentialScratch = nullptr;

    float* temperature = nullptr;       // T [K]

    // Future plasma fields
    float* electronDensity = nullptr; // ne [m^-3]
    float* electronTemperature = nullptr; // Te [eV] or K - choose explicitly

    // --- Vector fields ---
    float4* electricField = nullptr;    // E [V/m]
    float4* magneticField = nullptr;    // B [T]
    float4* currentDensity = nullptr;   // J [A/m^2]
};

//
// ============================================================
// FIELD VISUALIZATION
// ============================================================
//

// Render-only data.
// Must never affect physical field values.
//
struct FieldGlyphVertex {
    float4 position;
    float4 color;
};

// Visual settings only. Fixed slots are transparent below threshold; SI arrays
// are never scaled or clipped by these parameters. Stride applies per axis.
struct FieldRenderParams {
    unsigned stride = 1;
    unsigned vectorScale = 2; // 0 direction, 1 relative, 2 logarithmic
    double vectorThreshold = 0.0;
    double scalarThreshold = 0.0;
    double vectorReference = 1.44e-8;
    double scalarReference = 1.0e-18;
    double logStrength = 1000.0;
    float lengthInCells = 0.65f;
    float4 vectorColor = make_float4(0.2f, 0.9f, 1.0f, 0.85f);
};

static_assert(std::is_trivial<ParticleSimParams>::value, "CUDA constant POD");
static_assert(std::is_trivial<FieldGridParams>::value, "CUDA constant POD");
static_assert(std::is_trivial<FieldSolverParams>::value, "CUDA constant POD");

#endif
