#ifndef _PARTICLES_KERNEL_CUH_
#define _PARTICLES_KERNEL_CUH_

#include "paramsEM_kernel.cuh"
#include "particleFieldType.h"
#include "kernel_impl.cuh"

#include <stdio.h>
#include <math.h>
#include <vector_types.h>
#include <helper_math.h>

#if USE_TEX
texture<float4, 1, cudaReadModeElementType> oldPosTex;
texture<float4, 1, cudaReadModeElementType> oldVelTex;

texture<uint, 1, cudaReadModeElementType> gridParticleHashTex;
texture<uint, 1, cudaReadModeElementType> cellStartTex;
texture<uint, 1, cudaReadModeElementType> cellEndTex;
#endif

__global__
void calcHashD(
	uint* gridParticleHash,
	uint* gridParticleIndex,
	float4* pos,
	uint numParticles
);
__global__
void reorderDataAndFindCellStartD(
	uint* cellStart,
	uint* cellEnd,
	float4* sortedPos,
	float4* sortedVel,
	uint* gridParticleHash,
	uint* gridParticleIndex,
	float4* oldPos,
	float4* oldVel,
	uint numParticles
);
__global__
void collideD(
	float4* newVel,
	float4* oldPos,
	float4* oldVel,
	uint* gridParticleIndex,
	uint* cellStart,
	uint* cellEnd,
	uint numParticles
);
__global__
void calculate_forces(
	float4* d_b,
	float4* d_a,
	uint numParticles
);
///////////////////////////////
///////////////////////////////
__device__ int3 calcGridPos(float3 p);
__device__ uint calcGridHash(int3 gridPos);
///////////////////////////////
__device__
float3 bodyBodyInteractions(
	float4 bi,
	float4 bj,
	float3 ai
);
__device__
float3 collideSpheres(
	float4 posA,
	float4 posB,
	float4 velA,
	float4 velB,
	float attraction
);
__device__
float3 tile_calculation(
	float4 myPosition,
	float3 acc,
	uint tile,
	uint numParticles
);
__device__
float3 collideCell(
	int3 gridPos,
	uint index,
	float4 pos,
	float4 vel,
	float4* oldPos,
	float4* oldVel,
	uint* cellStart,
	uint* cellEnd
);
///-----------------------------------------------------------------------------------------
/// <PARTICLE SYSTEM KERNEL>
///-----------------------------------------------------------------------------------------
__global__
void calcHashD(
	uint* gridParticleHash,
	uint* gridParticleIndex,
	float4* pos,
	uint numParticles) {
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;
	volatile float4 p = pos[index];
	// get addres in grid
	int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
	uint hash = calcGridHash(gridPos);
	// store grid hash and particle index
	gridParticleHash[index] = hash;
	gridParticleIndex[index] = index;
}
__global__
void reorderDataAndFindCellStartD(
	uint* cellStart,
	uint* cellEnd,
	float4* sortedPos,
	float4* sortedVel,
	uint* gridParticleHash,
	uint* gridParticleIndex,
	float4* oldPos,
	float4* oldVel,
	uint numParticles) {

	extern __shared__ uint sharedHash[]; // blockSize + 1 elements
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	uint hash;
	// handle case when no. of particles not multiple of block size
	if (index < numParticles) {
		hash = gridParticleHash[index];
		// Load hash data into shared memory so that we can look
		// at neighboring particle's hash value without loading
		// two hash values per thread
		sharedHash[threadIdx.x + 1] = hash;

		if (index > 0 && threadIdx.x == 0) {
			// first thread in block must load neighbor particle hash
			sharedHash[0] = gridParticleHash[index - 1];
		}
	}

	__syncthreads();
	if (index < numParticles) {
		// If this particle has a different cell index to the previous
		// particle then it must be the first particle in the cell,
		// so store the index of this particle in the cell.
		// As it isn't the first particle, it must also be the cell end of
		// the previous particle's cell
		if (index == 0 || hash != sharedHash[threadIdx.x]) {

			cellStart[hash] = index;

			if (index > 0)
				cellEnd[sharedHash[threadIdx.x]] = index;
		}

		if (index == numParticles - 1)
			cellEnd[hash] = index + 1;

		// Now use the sorted index to reorder the pos and vel data
		uint sortedIndex = gridParticleIndex[index];
		// macro does either global read or texture fetch
		float4 pos = FETCH(oldPos, sortedIndex);
		float4 vel = FETCH(oldVel, sortedIndex);

		sortedPos[index] = pos;
		sortedVel[index] = vel;
	}
}
__global__
void collideD(
	float4* newVel,
	float4* oldPos,
	float4* oldVel,
	uint* gridParticleIndex,
	uint* cellStart,
	uint* cellEnd,
	uint numParticles) {

	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= numParticles)
		return;

	// read particle data from sorted arrays
	float4 pos = FETCH(oldPos, index);
	float4 vel = FETCH(oldVel, index);

	float3 v_new = make_float3(vel.x, vel.y, vel.z);
	// get address in grid
	int3 gridPos = calcGridPos(make_float3(pos.x, pos.y, pos.z));
	float3 force = make_float3(0.0f); // examine neighbouring cells
	for (int z = -1; z <= 1; z++) {
		for (int y = -1; y <= 1; y++) {
			for (int x = -1; x <= 1; x++) {

				int3 neighbourPos = gridPos +
					make_int3(x, y, z);

				force += collideCell(
					neighbourPos,
					index,
					pos,
					vel,
					oldPos,
					oldVel,
					cellStart,
					cellEnd
				);
			}
		}
	}

	uint originalIndex = gridParticleIndex[index];
	newVel[originalIndex] = make_float4(v_new + force, vel.w);
}
__global__
void calculate_forces(
	float4* d_b,
	float4* d_a,
	uint numParticles) {

	extern __shared__ float4 shPosition[];
	const uint body_id = blockIdx.x * blockDim.x + threadIdx.x;

	if (body_id >= numParticles)
		return;

	float4 myPosition = d_b[body_id];
	float3 acc = make_float3(0.0f, 0.0f, 0.0f);

	for (uint tile = 0; tile < gridDim.x; tile++) {

		uint idx = tile * blockDim.x + threadIdx.x;

		if (idx < numParticles)
			shPosition[threadIdx.x] = d_b[idx];
		else
			shPosition[threadIdx.x] = make_float4(0, 0, 0, 0);

		__syncthreads();
		acc = tile_calculation(
			myPosition,
			acc,
			tile,
			numParticles
		);
		__syncthreads();
	}

	d_a[body_id] = make_float4(acc.x, acc.y, acc.z, 0.0f);
}
///-----------------------------------------------------------------------------------------
/// </PARTICLE SYSTEM KERNEL>
///-----------------------------------------------------------------------------------------

/// <PARTICLE SYSTEM DEVICE FUNCTIONS>
///-----------------------------------------------------------------------------------------
// calculate position in uniform grid
__device__
int3 calcGridPos(float3 p) {
	int3 gridPos;
	gridPos.x = floor(
		(p.x - cParticleParams.worldOrigin.x) /
		cParticleParams.cellSize.x
	);
	gridPos.y = floor(
		(p.y - cParticleParams.worldOrigin.y) /
		cParticleParams.cellSize.y
	);
	gridPos.z = floor(
		(p.z - cParticleParams.worldOrigin.z) /
		cParticleParams.cellSize.z
	);
	return gridPos;
}
// calculate address in grid from position (clamping to edges)
__device__
uint calcGridHash(int3 gridPos) {
	gridPos.x = gridPos.x & (cParticleParams.gridSize.x - 1); // wrap grid, assumes size is power of 2
	gridPos.y = gridPos.y & (cParticleParams.gridSize.y - 1);
	gridPos.z = gridPos.z & (cParticleParams.gridSize.z - 1);
	return ((gridPos.z * cParticleParams.gridSize.y) * cParticleParams.gridSize.x) +
		(gridPos.y * cParticleParams.gridSize.x) + gridPos.x;
}
// calculate grid hash value for each particle
__device__
float3 bodyBodyInteractions(
	float4 bi,
	float4 bj,
	float3 ai) {
	float3 posi = make_float3(bi);
	float3 posj = make_float3(bj);
	float3 r = posj - posi;
	float distSqr =
		(r.x * r.x) +
		(r.y * r.y) +
		(r.z * r.z) + EPS2;
	float distSixth = distSqr * distSqr * distSqr;
	float invDistSqr = 1.0f / sqrtf(distSixth);
	float gm_r3 = G * bj.w * invDistSqr;
	ai.x += gm_r3 * r.x;
	ai.y += gm_r3 * r.y;
	ai.z += gm_r3 * r.z;
	return ai;
}
__device__
float3 tile_calculation(
	float4 myPosition,
	float3 acc,
	uint tile,
	uint numParticles) {
	extern __shared__ float4 shPosition[];
	for (uint j = 0; j < blockDim.x; j++) {
		uint other = tile * blockDim.x + j;

		if (other < numParticles) {

			acc = bodyBodyInteractions(
				myPosition,
				shPosition[j],
				acc
			);
		}
	}
	return acc;
}
__device__
float3 collideSpheres(
	float4 posA,
	float4 posB,
	float4 velA,
	float4 velB,
	float attraction) {
	const float3 pos_A = make_float3(posA);
	const float3 pos_B = make_float3(posB);
	const float3 vel_A = make_float3(velA);
	const float3 vel_B = make_float3(velB);
	float3 relPos = pos_B - pos_A;
	float dist = length(relPos);
	float collideDist = velA.w + velB.w;
	float3 force = make_float3(0.0f);
	if (dist < collideDist) {
		float3 norm = relPos / dist;
		float3 relVel = vel_B - vel_A;
		float3 tanVel = relVel - (dot(relVel, norm) * norm);
		// spring force
		force = -cParticleParams.spring * (collideDist - dist) * norm;
		// dashpot (damping) force
		force += cParticleParams.damping * relVel;
		// tangential shear force
		force += cParticleParams.shear * tanVel;
		// attraction
		force += cParticleParams.attraction * relPos;
	}
	return force;
}
__device__
float3 collideCell(
	int3 gridPos,
	uint index,
	float4 pos,
	float4 vel,
	float4* oldPos,
	float4* oldVel,
	uint* cellStart,
	uint* cellEnd) {
	uint gridHash = calcGridHash(gridPos);
	uint startIndex = FETCH(cellStart, gridHash);
	float3 force = make_float3(0.0f);
	if (startIndex != 0xffffffff) { // cell is not empty
		// iterate over particles in this cell
		uint endIndex = FETCH(cellEnd, gridHash);
		for (int j = startIndex; j < endIndex; j++) {
			if (j != index) {
				// check not colliding with self
				float4 pos2 = FETCH(oldPos, j);
				float4 vel2 = FETCH(oldVel, j);
				force += collideSpheres(pos, pos2, vel, vel2, cParticleParams.attraction);
			}
		}
	}
	return force;
}
///-----------------------------------------------------------------------------------------
/// </PARTICLE SYSTEM DEVICE FUNCS>
///-----------------------------------------------------------------------------------------

#endif