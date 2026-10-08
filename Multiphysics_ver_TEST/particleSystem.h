#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__

#define DEBUG_GRID 0
#define DO_TIMING 0

#include <helper_functions.h>
#include <vector_functions.h>

#include "paramsEM_kernel.cuh"
#include "particleFieldType.h"
class FieldSystem;

struct ParticleProxy3D {
	float4 position;
	float4 velocity;
	float4 acceleration;
	
	float radius = 0.0f;
	uint objectId = 0;
	
	ParticleProxy3D()
		: position(make_float4(0.0f, 0.0f, 0.0f, 1.0f)),
		velocity(make_float4(0.0f, 0.0f, 0.0f, 0.0f)),
		acceleration(make_float4(0.0f, 0.0f, 0.0f, 0.0f)),
		radius(0.0f),
		objectId(0) {
	}
};

class ParticleSystem {
public:
	
	ParticleSystem(
		uint numParticles,
		uint3 gridSize,
		bool bUseOpenGL
	);
	
	~ParticleSystem();
	
	enum ParticleConfig {
		CNFG_DEFAULT_RESTART,
		CNFG_RANDOM_RESTART,
		_NUM_CONFIGS
	};
	
	enum ParticleArray {
		POSITION,
		VELOCITY,
		ACCELERATION
	};
	
	enum ParticleClass {
		PARTICLE_NONE = 0,
		RED,
		BLUE,
		GREEN
	};
	
	int getNumParticles() const { return m_numParticles; }
	uint getCapacity() const { return m_numParticles; }
	uint getActiveParticleCount() const { return m_activeParticleCount; }
	bool setActiveParticleCount(uint count);
	unsigned int getCurrentReadBuffer() const { return m_posVbo; }
	unsigned int getRadiiBuffer() const { return m_radVbo; }
	unsigned int getColorBuffer() const { return m_colorVBO; }
	
	float getParticleRadius() { return m_params.particleRadius; }
	uint3 getGridSize() { return m_params.gridSize; }
	float3 getCellSize() { return m_params.cellSize; }
	float3 getWorldOrigin() { return m_params.worldOrigin; }
	
	float* getArray(ParticleArray array);
	float4 getSingleParticle(ParticleArray array, uint index);
	float4 getUniformParticleColor() const { return m_uniformParticleColor; }
	
	// --- UNDER CONSTRUCTION ---
	//void setSimBoundary(float x) { m_params.boundary = x; }
	void setSimulationDomain(float boxSize);
	// Reallocates only collision cell tables when the power-of-two grid changes.
	void setSimulationDomain(float boxSize, uint3 gridSize);
	uint getNumGridCells() const { return m_numGridCells; }
	// --- UNDER CONSTRUCTION ---

	void dumpGrid();
	void dumpRadii(float* rad);
	void dumpRadii(float* rad, uint count);
	void dumpParticles(uint start, uint count);
	
    void update(float deltaTime);
    bool updateMultiphysics(float deltaTime, FieldSystem& fields, ElectrostaticMode mode, double timeSeconds);
    bool setFieldMarkers(const ParticleFieldMarker* markers, unsigned count);
    bool setMetersPerWorldUnit(double scale);
    double metersPerWorldUnit() const { return m_params.metersPerWorldUnit; }
	void reset(ParticleConfig config);
	void setDefaultColorRamp();

	bool resetInBounds(
		ParticleConfig config,
		const float3& minimum,
		const float3& maximum,
		float placementRadius,
		uint seed = 1973
	);
	
	bool setRGBParticleCounts(
		uint redCount,
		uint greenCount,
		uint blueCount
	);

	bool setRandomActiveRadii(
		float minimumRadius,
		float maximumRadius,
		uint seed = 1973
	);

	bool setUniformActiveRadii(float radius);
	bool setActiveRadii(const float* radii, uint count);
	bool setActiveColors(const float4* colors, uint count);

	void setUniformParticleColor(float r, float g, float b, float a = 1.0f);
	void setParticle(ParticleArray array, int index, float* data);
	void setArray(ParticleArray array, const float* data, int start, int count);
	
	void setIterations(int i) { m_solverIterations = i; }
	void setDamping(float x) { m_params.globalDamping = x; }
	void setGravity(float x) { m_params.gravity = make_float3(0.0f, x, 0.0f); }
	void setParticleRadius(float x) { m_params.particleRadius = x; }
	void setCollideSpring(float x) { m_params.spring = x; }
	void setCollideDamping(float x) { m_params.damping = x; }
	void setCollideShear(float x) { m_params.shear = x; }
	void setCollideAttraction(float x) { m_params.attraction = x; }
	
	void* getCudaPosVBO() const { return (void*)m_cudaPosVBO; }
	void* getCudaColorVBO() const { return (void*)m_cudaColorVBO; }
	
	ParticleProxy3D getSingleParticleProxy(uint index = 0);
	ParticleProxy3D getActiveParticle() { return getSingleParticleProxy(0); }
	
	uint addSphere(
		uint start,
		const float* position,
		const float* velocity,
		int latticeRadius,
		float spacing
	);
	
protected:
	ParticleSystem() {}
	
	uint createVBO(uint size);
	
    bool updateInternal(float dt, FieldSystem* fields, ElectrostaticMode mode, double timeSeconds);
	void _initialize(uint numParticles);
	void _finalize();
	void initGrid(uint* size, float spacing, float jitter, uint numParticles);
	
protected:
	bool m_bInitialized;
	bool m_nbody = false;
	bool m_collisions = true;
	
	/// <CPU DATA> //////////////////////////////////////////////
	uint m_numParticles;
	uint m_activeParticleCount;
	uint m_numGridCells;
	
	uint* m_hParticleHash;
	uint* m_hCellStart;
	uint* m_hCellEnd;
	uint3 m_gridSize;
	
	float* m_hPos;
	float* m_hVel;
	float* m_hAcc;
	/// </CPU DATA> //////////////////////////////////////////////
	
	int m_solverIterations;
	
	/// <GPU DATA> //////////////////////////////////////////////
	uint m_radVbo;
	uint m_posVbo;
	uint m_colorVBO;
	uint m_gridSortBits;
	
	uint* m_dGridParticleHash;
	uint* m_dGridParticleIndex;
	uint* m_dCellStart;
	uint* m_dCellEnd;
	
	float* m_dPos;
	float* m_dVel;
	float* m_dAcc;
    ParticleFieldMarker* m_dMarkers = nullptr;
	float* m_dSortedPos;
	float* m_dSortedVel;
	float* m_cudaPosVBO;
	float* m_cudaColorVBO;
	
	struct cudaGraphicsResource* m_cuda_posvbo_resource;
	struct cudaGraphicsResource* m_cuda_radvbo_resource;
	struct cudaGraphicsResource* m_cuda_colorvbo_resource;
	/// </GPU DATA> //////////////////////////////////////////////
	
	ParticleSimParams m_params;
	ParticleClass* m_particleClass;
	
	float4 m_uniformParticleColor{ 1.0f, 0.05f, 0.0f, 1.0f };
};

#endif
