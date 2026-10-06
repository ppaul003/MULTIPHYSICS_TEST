#include "particleSystem.h"
#include "kernel.h"

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <assert.h>
#include <math.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <filesystem>

#include <GL/glew.h>

#ifndef CUDART_PI_F
#define CUDART_PI_F 3.141592654f
#endif

using namespace std;

namespace {

	inline float lerpColor(float a, float b, float t) {

		return a + t * (b - a);
	}

	void colorRamp(float t, float* color) {
		static constexpr int kColorCount = 7;

		static constexpr float colors[kColorCount][3] = {
			{ 1.0f, 0.0f, 0.0f },
			{ 1.0f, 0.5f, 0.0f },
			{ 1.0f, 1.0f, 0.0f },
			{ 0.0f, 1.0f, 0.0f },
			{ 0.0f, 1.0f, 1.0f },
			{ 0.0f, 0.0f, 1.0f },
			{ 1.0f, 0.0f, 1.0f }
		};

		t = (std::max)(0.0f, (std::min)(1.0f, t));

		const float scaled =
			t * static_cast<float>(kColorCount - 1);

		int index =
			static_cast<int>(floor(scaled));

		// color[index + 1] must remain valid,
		// including when t == 1.0.
		index = (std::min)(index, kColorCount - 2);

		const float localT =
			scaled -
			static_cast<float>(index);

		color[0] =
			lerpColor(
				colors[index][0],
				colors[index + 1][0],
				localT
			);

		color[1] =
			lerpColor(
				colors[index][1],
				colors[index + 1][1],
				localT
			);

		color[2] =
			lerpColor(
				colors[index][2],
				colors[index + 1][2],
				localT
			);

	}
}

ParticleSystem::ParticleSystem(
	uint numParticles,
	uint3 gridSize,
	bool bUseOpenGL) :
	m_bInitialized(false),
	m_numParticles(numParticles),
	m_activeParticleCount(numParticles),
	m_numGridCells(0),
	m_hParticleHash(0),
	m_hCellStart(0),
	m_hCellEnd(0),
	m_gridSize(gridSize),
	m_hPos(0),
	m_hVel(0),
	m_hAcc(0),
	m_solverIterations(1),
	m_dPos(0),
	m_dVel(0),
	m_dAcc(0),
	m_radVbo(0),
	m_posVbo(0),
	m_colorVBO(0),
	m_cuda_posvbo_resource(nullptr),
	m_cuda_radvbo_resource(nullptr),
	m_cuda_colorvbo_resource(nullptr),
	m_gridSortBits(0),
	m_dGridParticleHash(nullptr),
	m_dGridParticleIndex(nullptr),
	m_dCellStart(nullptr),
	m_dCellEnd(nullptr),
	m_dSortedPos(nullptr),
	m_dSortedVel(nullptr),
	m_cudaPosVBO(nullptr),
	m_cudaColorVBO(nullptr) {
	
	m_numGridCells =
		m_gridSize.x *
		m_gridSize.y *
		m_gridSize.z;
	
	m_gridSortBits = 18;
	m_params.gridSize = m_gridSize;
	m_params.numCells = m_numGridCells;
	m_params.numBodies = m_activeParticleCount;
	
	m_params.maxParticlesPerCell = 0;
	m_params.particleRadius = 0.012f;
	
	float cellSize = m_params.particleRadius * 2.0f;
	m_params.gravity = make_float3(0.0f, 0.0f, 0.0f);
	m_params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
	m_params.cellSize = make_float3(cellSize, cellSize, cellSize);
	
	m_params.shear = 0.1f;
	m_params.spring = 0.5f;
	m_params.damping = 0.02f;
	m_params.boundary = 1.0f;
	m_params.attraction = 0.0f;
	m_params.globalDamping = 1.0f;
	m_params.boundaryDamping = -0.5f;
	
	_initialize(numParticles);
}

ParticleSystem::~ParticleSystem() {
	_finalize();
	m_activeParticleCount = 0;
	m_numParticles = 0;
}

uint ParticleSystem::createVBO(uint size) {
	
	GLuint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	
	return vbo;
}



bool ParticleSystem::setActiveColors(const float4* colors, uint count) {
	if (!m_bInitialized || 
		!colors || 
		count != m_activeParticleCount ||
		m_colorVBO == 0) return false;

	if (count == 0) 
		return true;

	glBindBuffer(GL_ARRAY_BUFFER, m_colorVBO);

	glBufferSubData(
		GL_ARRAY_BUFFER, 
		0,
		static_cast<GLsizeiptr>(count * sizeof(float4)),
		colors
	);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	return true;
}

void ParticleSystem::_initialize(uint numParticles) {
	assert(!m_bInitialized);
	
	m_numParticles = static_cast<uint>(numParticles);
	
	// ALLOCATE GPU DATA
	unsigned int cSize = sizeof(uint) * m_numGridCells;
	unsigned int uSize = sizeof(uint) * m_numParticles;
	unsigned int mSize = sizeof(float) * m_numParticles;
	unsigned int memSize = sizeof(float) * 4 * m_numParticles;
	
	// ALLOCATE HOST STORAGE
	m_hPos = new float[m_numParticles * 4];
	m_hVel = new float[m_numParticles * 4];
	m_hAcc = new float[m_numParticles * 4];
	memset(m_hPos, 0, memSize);
	memset(m_hVel, 0, memSize);
	memset(m_hAcc, 0, memSize);
	
	m_hParticleHash = new uint[m_numParticles];
	m_hCellStart = new uint[m_numGridCells];
	m_hCellEnd = new uint[m_numGridCells];
	memset(m_hParticleHash, 0, uSize);
	memset(m_hCellStart, 0, cSize);
	memset(m_hCellEnd, 0, cSize);
	
	m_particleClass = new ParticleClass[m_numParticles];
	memset(m_particleClass, 0, m_numParticles * sizeof(ParticleClass));
	if (m_numParticles > 0) m_particleClass[0] = RED;
	if (m_numParticles > 1) m_particleClass[1] = GREEN;
	if (m_numParticles > 2) m_particleClass[2] = BLUE;
	
	m_radVbo = createVBO(mSize);
	registerGLBufferObject(m_radVbo, &m_cuda_radvbo_resource);
	
	m_posVbo = createVBO(memSize);
	registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
	
	allocateArray((void**)&m_dVel, memSize);
	allocateArray((void**)&m_dAcc, memSize);
	allocateArray((void**)&m_dSortedPos, memSize);
	allocateArray((void**)&m_dSortedVel, memSize);
	allocateArray((void**)&m_dGridParticleHash, uSize);
	allocateArray((void**)&m_dGridParticleIndex, uSize);
	allocateArray((void**)&m_dCellStart, cSize);
	allocateArray((void**)&m_dCellEnd, cSize);
	
	m_colorVBO = createVBO(memSize);
	registerGLBufferObject(m_colorVBO, &m_cuda_colorvbo_resource);
	
	glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO);
	float* data = (float*)glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	
	float* ptr = data;
	for (uint i = 0; i < numParticles; i++) {
		switch (m_particleClass[i]) {
		case GREEN: ptr[0] = 0.0f; ptr[1] = 1.0f; ptr[2] = 0.0f; break; // green
		case BLUE: ptr[0] = 0.0f; ptr[1] = 0.0f; ptr[2] = 1.0f; break; // blue
		case RED: ptr[0] = 1.0f; ptr[1] = 0.0f; ptr[2] = 0.0f; break; // red
		default: ptr[0] = 0.5f; ptr[1] = 0.5f; ptr[2] = 0.5f; break; // gray
		}
		
		ptr += 3;
		*ptr++ = 1.0f;
	}
	
	glUnmapBufferARB(GL_ARRAY_BUFFER);
	glBindBufferARB(GL_ARRAY_BUFFER, 0);
	
	setParameters(&m_params);
	
	m_bInitialized = true;
}

void ParticleSystem::_finalize() {
	assert(m_bInitialized);

	delete[] m_hPos;
	delete[] m_hVel;
	delete[] m_hAcc;
	delete[] m_hParticleHash;
	delete[] m_hCellStart;
	delete[] m_hCellEnd;
	delete[] m_particleClass;

	freeArray(m_dVel);
	freeArray(m_dAcc);
	freeArray(m_dSortedPos);
	freeArray(m_dSortedVel);
	freeArray(m_dGridParticleHash);
	freeArray(m_dGridParticleIndex);
	freeArray(m_dCellStart);
	freeArray(m_dCellEnd);

	unregisterGLBufferObject(m_cuda_colorvbo_resource);
	unregisterGLBufferObject(m_cuda_posvbo_resource);
	unregisterGLBufferObject(m_cuda_radvbo_resource);

	glDeleteBuffers(1, (const GLuint*)&m_colorVBO);
	glDeleteBuffers(1, (const GLuint*)&m_posVbo);
	glDeleteBuffers(1, (const GLuint*)&m_radVbo);

}

void ParticleSystem::setSimulationDomain(float boxSize) {
	if (boxSize <= 0.0f) return;

	const float halfBox =
		boxSize * 0.5f;

	m_params.boundary = halfBox;

	m_params.worldOrigin =
		make_float3(-halfBox, -halfBox, -halfBox);

	m_params.cellSize =
		make_float3(
			boxSize / static_cast<float>(m_params.gridSize.x),
			boxSize / static_cast<float>(m_params.gridSize.y),
			boxSize / static_cast<float>(m_params.gridSize.z)
		);
}

bool ParticleSystem::setActiveParticleCount(uint count) {
	if (count > m_numParticles) return false;

	m_activeParticleCount = count;
	m_params.numBodies = count;

	if (m_bInitialized)
		setParameters(&m_params);

	return true;
}

void ParticleSystem::update(float deltaTime) {
	assert(m_bInitialized);

	if (m_activeParticleCount == 0) return;

	float* dPos = (float*)mapGLBufferObject(&m_cuda_posvbo_resource);

	setParameters(&m_params);
	integrateSystem(
		dPos,
		m_dVel,
		m_dAcc,
		deltaTime,
		m_activeParticleCount
	);

	if (m_nbody)
		forcesKernel(dPos, m_dAcc, m_activeParticleCount);

	if (m_collisions) {
		calcHash(
			m_dGridParticleHash,
			m_dGridParticleIndex,
			dPos,
			m_activeParticleCount
		);

		sortParticles(
			m_dGridParticleHash,
			m_dGridParticleIndex,
			m_activeParticleCount
		);

		reorderDataAndFindCellStart(
			m_dCellStart,
			m_dCellEnd,
			m_dSortedPos,
			m_dSortedVel,
			m_dGridParticleHash,
			m_dGridParticleIndex,
			dPos,
			m_dVel,
			m_activeParticleCount,
			m_numGridCells
		);

		collide(
			m_dVel,
			m_dSortedPos,
			m_dSortedVel,
			m_dGridParticleIndex,
			m_dCellStart,
			m_dCellEnd,
			m_activeParticleCount,
			m_numGridCells
		);
	}

	unmapGLBufferObject(m_cuda_posvbo_resource);
}

void ParticleSystem::dumpGrid() {
	// dump grid information

	copyArrayFromDevice(m_hCellStart, m_dCellStart, 0, sizeof(uint) * m_numGridCells);
	copyArrayFromDevice(m_hCellEnd, m_dCellEnd, 0, sizeof(uint) * m_numGridCells);

	uint maxCellSize = 0;
	for (uint i = 0; i < m_numGridCells; i++) {
		if (m_hCellStart[i] != 0xffffffff) {

			uint cellSize = 
				m_hCellEnd[i] - m_hCellStart[i];

			// printf("cell: %d, %d particles\n", i, cellSize);
			if (cellSize > maxCellSize) maxCellSize = cellSize;
		}
	}

	printf("maximum particles per cell = %d\n", maxCellSize);
}

void ParticleSystem::dumpParticles(uint start, uint count) {
	// debug

	copyArrayFromDevice(m_hPos, 0, &m_cuda_posvbo_resource, sizeof(float) * 4 * count);
	copyArrayFromDevice(m_hVel, m_dVel, 0, sizeof(float) * 4 * count);
	for (uint i = start; i < start + count; i++) {
		//        printf("%d: ", i);
		printf("pos: (%.4f, %.4f, %.4f, %.4f)\n", m_hPos[i * 4 + 0], m_hPos[i * 4 + 1], m_hPos[i * 4 + 2], m_hPos[i * 4 + 3]);
		printf("vel: (%.4f, %.4f, %.4f, %.4f)\n", m_hVel[i * 4 + 0], m_hVel[i * 4 + 1], m_hVel[i * 4 + 2], m_hVel[i * 4 + 3]);
	}
}

void ParticleSystem::dumpRadii(float* rad) {
	dumpRadii(rad, m_activeParticleCount);
}

void ParticleSystem::dumpRadii(float* rad, uint count) {
	if (!rad ||
		count > m_numParticles ||
		count > m_activeParticleCount ||
		count == 0) return;

	copyArrayFromDevice(
		m_hVel,
		m_dVel,
		0,
		sizeof(float) * 4 * count
	);

	for (uint i = 0; i < count; i++) {
		rad[i] = m_hVel[i * 4 + 3];
	}
}

bool ParticleSystem::setActiveRadii(
	const float* radii, 
	uint count) {

	static constexpr float kMaximumSupportedRadius = 0.0156f;

	if (!m_bInitialized || 
		!radii || 
		count != m_activeParticleCount)
		return false;

	if (count == 0) 
		return true;

	float maximumRadius = 0.0f;

	for (uint i = 0; i < count; i++) {
		if (!isfinite(radii[i]) ||
			radii[i] <= 0.0f ||
			radii[i] > kMaximumSupportedRadius)
			return false;

		maximumRadius = std::max(maximumRadius, radii[i]);
	}

	copyArrayFromDevice(
		m_hVel, 
		m_dVel, 
		0, 
		sizeof(float) * 4 * count
	);

	for (uint i = 0; i < count; i++)
		m_hVel[i * 4 + 3] = radii[i];

	// Conservative scalar used by existing
	// collision/grid code
	m_params.particleRadius = maximumRadius;

	setParameters(&m_params);

	setArray(
		VELOCITY, 
		m_hVel, 
		0, 
		static_cast<int>(count)
	);

	return true;
}

bool ParticleSystem::setUniformActiveRadii(float radius) {
	static constexpr float kMaximumSupportedRadius = 0.0156f;

	if (!m_bInitialized ||
		!std::isfinite(radius) ||
		radius <= 0.0f ||
		radius > kMaximumSupportedRadius) {

		return false;
	}

	m_params.particleRadius = radius;
	setParameters(&m_params);

	if (m_activeParticleCount == 0)
		return true;

	copyArrayFromDevice(
		m_hVel,
		m_dVel,
		0,
		sizeof(float) * 4 * m_activeParticleCount
	);

	for (uint i = 0; i < m_activeParticleCount; ++i)
		m_hVel[i * 4 + 3] = radius;

	setArray(
		VELOCITY,
		m_hVel,
		0,
		static_cast<int>(m_activeParticleCount)
	);

	return true;
}

bool ParticleSystem::setRandomActiveRadii(
	float minimumRadius,
	float maximumRadius,
	uint seed) {

	static constexpr float kMaximumSupportedRadius = 0.0156f;

	if (!m_bInitialized ||
		!std::isfinite(minimumRadius) ||
		!std::isfinite(maximumRadius) ||
		minimumRadius <= 0.0f ||
		minimumRadius > maximumRadius ||
		maximumRadius > kMaximumSupportedRadius) {

		return false;
	}

	// The maximum radius is the conservative scalar reference used by
	// placement and renderer fallbacks. Each active particle still owns its
	// exact collision/render radius in velocity.w.
	m_params.particleRadius = maximumRadius;
	setParameters(&m_params);

	if (m_activeParticleCount == 0)
		return true;

	copyArrayFromDevice(
		m_hVel,
		m_dVel,
		0,
		sizeof(float) * 4 * m_activeParticleCount
	);

	std::mt19937 generator(seed);
	std::uniform_real_distribution<float> radiusDistribution(
		minimumRadius,
		maximumRadius
	);

	for (uint i = 0; i < m_activeParticleCount; ++i)
		m_hVel[i * 4 + 3] = radiusDistribution(generator);

	setArray(
		VELOCITY,
		m_hVel,
		0,
		static_cast<int>(m_activeParticleCount)
	);

	return true;
}

float* ParticleSystem::getArray(ParticleArray array) {
	assert(m_bInitialized);

	float* hdata = 0;
	float* ddata = 0;
	struct cudaGraphicsResource* cuda_vbo_resource = 0;

	switch (array) {
	default:
	case POSITION:
		hdata = m_hPos;
		ddata = m_dPos;
		cuda_vbo_resource = m_cuda_posvbo_resource;
		break;

	case VELOCITY:
		hdata = m_hVel;
		ddata = m_dVel;
		break;

	case ACCELERATION:
		hdata = m_hAcc;
		ddata = m_dAcc;
		break;
	}

	unsigned int memSize = sizeof(float) * 4 * m_numParticles;
	copyArrayFromDevice(hdata, ddata, &cuda_vbo_resource, memSize);

	return hdata;
}

float4 ParticleSystem::
getSingleParticle(ParticleArray array, uint index) {

	if (!m_bInitialized || index >= m_numParticles) {
		return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	}

	const unsigned int memSize =
		sizeof(float) * 4 * m_numParticles;

	switch (array) {
	default:
	case POSITION:
		copyArrayFromDevice(
			m_hPos,
			nullptr,
			&m_cuda_posvbo_resource,
			memSize
		);

		return make_float4(
			m_hPos[index * 4 + 0],
			m_hPos[index * 4 + 1],
			m_hPos[index * 4 + 2],
			m_hPos[index * 4 + 3]
		);

		break;

	case VELOCITY:

		copyArrayFromDevice(
			m_hVel,
			m_dVel,
			nullptr,
			memSize
		);

		return make_float4(
			m_hVel[index * 4 + 0],
			m_hVel[index * 4 + 1],
			m_hVel[index * 4 + 2],
			m_hVel[index * 4 + 3]
		);


		break;

	case ACCELERATION:

		copyArrayFromDevice(
			m_hAcc,
			m_dAcc,
			nullptr,
			memSize
		);

		return make_float4(
			m_hAcc[index * 4 + 0],
			m_hAcc[index * 4 + 1],
			m_hAcc[index * 4 + 2],
			m_hAcc[index * 4 + 3]
		);
		break;
	}
}

void ParticleSystem::setParticle(ParticleArray array, int index, float* data) {
	assert(m_bInitialized);

	if (!data) return;
	if (index < 0 || static_cast<uint>(index) >= m_numParticles) return;

	float* host = nullptr;
	switch (array) {
	default:
	case POSITION:
		host = m_hPos;
		break;

	case VELOCITY:
		host = m_hVel;
		break;

	case ACCELERATION:
		host = m_hAcc;
		break;
	}

	const int base = index * 4;
	for (int c = 0; c < 4; ++c) {
		host[base + c] = data[c];
	}

	setArray(array, &host[base], index, 1);
}

void ParticleSystem::setArray(ParticleArray array, const float* data, int start, int count) {
	assert(m_bInitialized);

	switch (array) {
	default:
	case POSITION: {
		unregisterGLBufferObject(m_cuda_posvbo_resource);
		glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
		glBufferSubData(GL_ARRAY_BUFFER, start * 4 * sizeof(float), count * 4 * sizeof(float), data);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
	}
				break;

	case VELOCITY:
		copyArrayToDevice(m_dVel, data, start * 4 * sizeof(float), count * 4 * sizeof(float));
		break;

	case ACCELERATION:
		copyArrayToDevice(m_dAcc, data, start * 4 * sizeof(float), count * 4 * sizeof(float));
		break;
	}
}

void ParticleSystem::setDefaultColorRamp() {

	if (!m_bInitialized ||
		m_colorVBO == 0 ||
		m_activeParticleCount == 0) return;

	vector<float> colors(
		m_activeParticleCount * 4,
		1.0f
	);

	for (uint i = 0; i < m_activeParticleCount; i++) {

		const float t =
			m_activeParticleCount > 1
			? static_cast<float>(i) /
				static_cast<float>(m_activeParticleCount - 1)
			: 0.0f;

		float* particleColor =
			colors.data() + i * 4;

		colorRamp(t, particleColor);

		particleColor[3] = 1.0f;
		m_particleClass[i] = PARTICLE_NONE;
	}

	glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO);

	glBufferSubData(
		GL_ARRAY_BUFFER,
		0,
		static_cast<GLsizeiptr>(colors.size() * sizeof(float)),
		colors.data()
	);

	glBindBufferARB(GL_ARRAY_BUFFER, 0);
}

bool ParticleSystem::setRGBParticleCounts(
	uint redCount,
	uint greenCount,
	uint blueCount) {

	const unsigned long long total =
		static_cast<unsigned long long>(redCount) +
		static_cast<unsigned long long>(greenCount) +
		static_cast<unsigned long long>(blueCount);

	if (total > m_numParticles ||
		total != m_activeParticleCount)
		return false;

	if (!m_bInitialized || m_colorVBO == 0)
		return false;

	if (m_activeParticleCount == 0)
		return true;

	vector<float> colors(
		m_activeParticleCount * 4,
		1.0f
	);

	for (uint i = 0; i < m_activeParticleCount; i++) {
		float r = 0.0f;
		float g = 0.0f;
		float b = 0.0f;
		ParticleClass particleClass = PARTICLE_NONE;

		if (i < redCount) {
			r = 1.00f;
			g = 0.05f;
			b = 0.00f;
			particleClass = RED;
		}
		else if (i < redCount + greenCount) {
			r = 0.00f;
			g = 1.00f;
			b = 0.25f;
			particleClass = GREEN;
		}
		else {
			r = 0.00f;
			g = 0.25f;
			b = 1.00f;
			particleClass = BLUE;
		}

		colors[i * 4 + 0] = r;
		colors[i * 4 + 1] = g;
		colors[i * 4 + 2] = b;
		colors[i * 4 + 3] = 1.0f;
		m_particleClass[i] = particleClass;
	}

	glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO);
	glBufferSubData(
		GL_ARRAY_BUFFER,
		0,
		static_cast<GLsizeiptr>(colors.size() * sizeof(float)),
		colors.data()
	);
	glBindBufferARB(GL_ARRAY_BUFFER, 0);

	return true;
}

void ParticleSystem::setUniformParticleColor(float r, float g, float b, float a) {
	m_uniformParticleColor = make_float4(r, g, b, a);

	if (!m_bInitialized || m_colorVBO == 0) return;

	vector<float> colors(m_numParticles * 4);
	for (uint i = 0; i < m_numParticles; i++) {
		colors[i * 4 + 0] = r;
		colors[i * 4 + 1] = g;
		colors[i * 4 + 2] = b;
		colors[i * 4 + 3] = a;
	}

	glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO);
	glBufferSubData(
		GL_ARRAY_BUFFER,
		0,
		static_cast<GLsizeiptr>(colors.size() * sizeof(float)),
		colors.data()
	);
	glBindBufferARB(GL_ARRAY_BUFFER, 0);
}

inline float frand() {
	return rand() / (float)RAND_MAX;
}

void ParticleSystem::initGrid(uint* size, float spacing, float jitter, uint numParticles) {
	srand(1973);
	for (uint z = 0; z < size[2]; z++) {
		for (uint y = 0; y < size[1]; y++) {
			for (uint x = 0; x < size[0]; x++) {

				uint i = (z * size[1] * size[0]) + (y * size[0]) + x;

				if (i < numParticles) {
					m_hPos[i * 4] =
						m_params.worldOrigin.x +
						m_params.particleRadius +
						(spacing * x) +
						(frand() * 2.0f - 1.0f) * jitter;

					m_hPos[i * 4 + 1] =
						m_params.worldOrigin.y +
						m_params.particleRadius +
						(spacing * y) +
						(frand() * 2.0f - 1.0f) * jitter;

					m_hPos[i * 4 + 2] =
						m_params.worldOrigin.z +
						m_params.particleRadius +
						(spacing * z) +
						(frand() * 2.0f - 1.0f) * jitter;
					m_hPos[i * 4 + 3] = 1.0f;

					m_hVel[i * 4] = 0.0f;
					m_hVel[i * 4 + 1] = 0.0f;
					m_hVel[i * 4 + 2] = 0.0f;
					m_hVel[i * 4 + 3] = m_params.particleRadius;

					m_hAcc[i * 4] = 0.0f;
					m_hAcc[i * 4 + 1] = 0.0f;
					m_hAcc[i * 4 + 2] = 0.0f;
					m_hAcc[i * 4 + 3] = 0.0f;
				}
			}
		}
	}
}
void ParticleSystem::reset(ParticleConfig config) {
	const uint activeCount = m_activeParticleCount;
	m_params.numBodies = activeCount;

	if (activeCount == 0) return;

	switch (config) {
	default:
	case CNFG_DEFAULT_RESTART: {
		float jitter = m_params.particleRadius * 0.01f;
		uint s = static_cast<uint>(
			ceilf(powf(
				static_cast<float>(activeCount),
				1.0f / 3.0f
			))
		);
		uint gridSize[3];
		gridSize[0] = gridSize[1] = gridSize[2] = s;
		initGrid(
			gridSize,
			m_params.particleRadius * 2.0f,
			jitter,
			activeCount
		);
	}
							break;

	case CNFG_RANDOM_RESTART: {
		int p = 0, v = 0, a = 0;

		const float minimumX =
			m_params.worldOrigin.x +
			m_params.particleRadius;

		const float minimumY =
			m_params.worldOrigin.y +
			m_params.particleRadius;

		const float minimumZ =
			m_params.worldOrigin.z +
			m_params.particleRadius;

		const float maximumX =
			m_params.worldOrigin.x +
			m_params.cellSize.x * m_params.gridSize.x -
			m_params.particleRadius;

		const float maximumY =
			m_params.worldOrigin.y +
			m_params.cellSize.y * m_params.gridSize.y -
			m_params.particleRadius;

		const float maximumZ =
			m_params.worldOrigin.z +
			m_params.cellSize.z * m_params.gridSize.z -
			m_params.particleRadius;

		for (uint i = 0; i < activeCount; i++) {

			float point[3];
			point[0] = frand();
			point[1] = frand();
			point[2] = frand();

			m_hPos[p++] =
				minimumX +
				point[0] * (maximumX - minimumX);

			m_hPos[p++] =
				minimumY +
				point[1] * (maximumY - minimumY);

			m_hPos[p++] =
				minimumZ +
				point[2] * (maximumZ - minimumZ);
			m_hPos[p++] = 1.0f;

			m_hVel[v++] = 0.0f;
			m_hVel[v++] = 0.0f;
			m_hVel[v++] = 0.0f;
			m_hVel[v++] = m_params.particleRadius;

			m_hAcc[a++] = 0.0f;
			m_hAcc[a++] = 0.0f;
			m_hAcc[a++] = 0.0f;
			m_hAcc[a++] = 0.0f;
		}
	}
							break;
	}

	setArray(POSITION, m_hPos, 0, activeCount);
	setArray(VELOCITY, m_hVel, 0, activeCount);
	setArray(ACCELERATION, m_hAcc, 0, activeCount);
}

bool ParticleSystem::resetInBounds(
	ParticleConfig config,
	const float3& minimum,
	const float3& maximum,
	float placementRadius,
	uint seed) {

	if (!m_bInitialized ||
		config < CNFG_DEFAULT_RESTART ||
		config >= _NUM_CONFIGS ||
		!std::isfinite(minimum.x) ||
		!std::isfinite(minimum.y) ||
		!std::isfinite(minimum.z) ||
		!std::isfinite(maximum.x) ||
		!std::isfinite(maximum.y) ||
		!std::isfinite(maximum.z) ||
		!std::isfinite(placementRadius) ||
		placementRadius <= 0.0f) {
		return false;
	}

	const float3 lower = make_float3(
		minimum.x + placementRadius,
		minimum.y + placementRadius,
		minimum.z + placementRadius
	);
	const float3 upper = make_float3(
		maximum.x - placementRadius,
		maximum.y - placementRadius,
		maximum.z - placementRadius
	);

	if (lower.x > upper.x || lower.y > upper.y || lower.z > upper.z) {
		return false;
	}

	const uint activeCount = m_activeParticleCount;
	m_params.particleRadius = placementRadius;
	m_params.numBodies = activeCount;
	setParameters(&m_params);

	if (activeCount == 0) {
		return true;
	}

	if (config == CNFG_DEFAULT_RESTART) {
		const uint side = static_cast<uint>(
			ceilf(cbrtf(static_cast<float>(activeCount)))
		);

		for (uint i = 0; i < activeCount; ++i) {
			const uint x = i % side;
			const uint y = (i / side) % side;
			const uint z = i / (side * side);

			const float fx = (static_cast<float>(x) + 0.5f) /
				static_cast<float>(side);
			const float fy = (static_cast<float>(y) + 0.5f) /
				static_cast<float>(side);
			const float fz = (static_cast<float>(z) + 0.5f) /
				static_cast<float>(side);

			m_hPos[i * 4] = lower.x + fx * (upper.x - lower.x);
			m_hPos[i * 4 + 1] = lower.y + fy * (upper.y - lower.y);
			m_hPos[i * 4 + 2] = lower.z + fz * (upper.z - lower.z);
			m_hPos[i * 4 + 3] = 1.0f;
		}
	}
	else {
		std::mt19937 generator(seed);
		std::uniform_real_distribution<float> distributionX(lower.x, upper.x);
		std::uniform_real_distribution<float> distributionY(lower.y, upper.y);
		std::uniform_real_distribution<float> distributionZ(lower.z, upper.z);

		for (uint i = 0; i < activeCount; ++i) {
			m_hPos[i * 4] = distributionX(generator);
			m_hPos[i * 4 + 1] = distributionY(generator);
			m_hPos[i * 4 + 2] = distributionZ(generator);
			m_hPos[i * 4 + 3] = 1.0f;
		}
	}

	for (uint i = 0; i < activeCount; ++i) {
		m_hVel[i * 4] = 0.0f;
		m_hVel[i * 4 + 1] = 0.0f;
		m_hVel[i * 4 + 2] = 0.0f;
		m_hVel[i * 4 + 3] = placementRadius;

		m_hAcc[i * 4] = 0.0f;
		m_hAcc[i * 4 + 1] = 0.0f;
		m_hAcc[i * 4 + 2] = 0.0f;
		m_hAcc[i * 4 + 3] = 0.0f;
	}

	setArray(POSITION, m_hPos, 0, activeCount);
	setArray(VELOCITY, m_hVel, 0, activeCount);
	setArray(ACCELERATION, m_hAcc, 0, activeCount);
	return true;
}
ParticleProxy3D ParticleSystem::getSingleParticleProxy(uint index) {
	ParticleProxy3D p;

	if (!m_bInitialized ||
		index >= m_numParticles) return p;

	p.position =
		getSingleParticle(POSITION, index);

	p.velocity =
		getSingleParticle(VELOCITY, index);

	p.acceleration =
		getSingleParticle(ACCELERATION, index);

	p.radius =
		(p.velocity.w > 0.0f)
		? p.velocity.w
		: m_params.particleRadius;

	p.objectId = 0;

	return p;
}

uint ParticleSystem::addSphere(
	uint start,
	const float* position,
	const float* velocity,
	int latticeRadius,
	float spacing) {
	
	if (!m_bInitialized ||
		!position ||
		!velocity ||
		start >= m_numParticles ||
		latticeRadius < 0 ||
		spacing <= 0.0f) return 0;
	
	uint index = start;
	
	const float jitter =
		m_params.particleRadius * 0.01f;
	
	// A lattice radius of r extends approximately r spacing units
	// outward from the supplied center.
	
	const float sphereRadius =
		spacing * static_cast<float>(latticeRadius);
	
	for (int z = -latticeRadius;
		z <= latticeRadius && index < m_numParticles; z++) {
		
		for (int y = -latticeRadius;
			y <= latticeRadius && index < m_numParticles; y++) {
			
			for (int x = -latticeRadius;
				x <= latticeRadius && index < m_numParticles; x++) {
				
				const float dx = static_cast<float>(x) * spacing;
				const float dy = static_cast<float>(y) * spacing;
				const float dz = static_cast<float>(z) * spacing;
				
				const float distance =
					sqrtf(dx * dx + dy * dy + dz * dz);
				
				if (distance > sphereRadius) continue;
				
				const uint base = index * 4;
				
				m_hPos[base + 0] =
					position[0] + dx + (frand() * 2.0f - 1.0f) * jitter;
				
				m_hPos[base + 1] =
					position[1] + dy + (frand() * 2.0f - 1.0f) * jitter;
				
				m_hPos[base + 2] =
					position[2] + dz + (frand() * 2.0f - 1.0f) * jitter;
				// Current VitruGen position convention.
				m_hPos[base + 3] = 1.0f; // mass
				
				m_hVel[base + 0] = velocity[0];
				m_hVel[base + 1] = velocity[1];
				m_hVel[base + 2] = velocity[2];
				// Current VitruGen radius convention.
				m_hVel[base + 3] =
					velocity[3] > 0.0f
					? velocity[3]
					: m_params.particleRadius;
				
				m_hAcc[base + 0] = 0.0f;
				m_hAcc[base + 1] = 0.0f;
				m_hAcc[base + 2] = 0.0f;
				m_hAcc[base + 3] = 0.0f;
				
				index++;
			}
		}
	}
	
	const uint writtenCount = index - start;
	if (writtenCount == 0) return 0;
	
	setArray(
		POSITION,
		m_hPos + start * 4,
		static_cast<int>(start),
		static_cast<int>(writtenCount)
	);
	
	setArray(
		VELOCITY,
		m_hVel + start * 4,
		static_cast<int>(start),
		static_cast<int>(writtenCount)
	);
	
	setArray(
		ACCELERATION,
		m_hAcc + start * 4,
		static_cast<int>(start),
		static_cast<int>(writtenCount)
	);
	
	return writtenCount;
}
