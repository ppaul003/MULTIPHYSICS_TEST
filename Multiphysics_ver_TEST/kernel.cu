#include <GL/freeglut.h>

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include <helper_functions.h>
#include <helper_math.h>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

#include "kernel.h"
#include "kernel_impl.cuh"

#define TX_2D 32
#define TY_2D 32

extern "C" {
	void cudaInit(int argc, char** argv) {
		int devID;

		devID = findCudaDevice(argc, (const char**)argv);

		if (devID < 0) {

			printf("No CUDA Capable devices found, exiting...\n");
			exit(EXIT_SUCCESS);
		}
	}
	void cudaGLInit(int argc, char** argv) {
		findCudaGLDevice(argc, (const char**)argv);
	}

	void allocateArray(void** devPtr, size_t size) {
		cudaMalloc(devPtr, size);
	}

	void freeArray(void* devPtr) {
		cudaFree(devPtr);
	}

	void threadSync() {
		cudaDeviceSynchronize();
	}

	void copyArrayToDevice(void* device, const void* host, int offset, int size) {
		cudaMemcpy((char*)device + offset, host, size, cudaMemcpyHostToDevice);
	}

	void registerGLBufferObject(uint vbo, struct cudaGraphicsResource** cuda_vbo_resource) {
		cudaGraphicsGLRegisterBuffer(cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone);
	}

	void unregisterGLBufferObject(struct cudaGraphicsResource* cuda_vbo_resource) {
		cudaGraphicsUnregisterResource(cuda_vbo_resource);
	}

	void* mapGLBufferObject(struct cudaGraphicsResource** cuda_vbo_resource) {

		void* ptr;
		cudaGraphicsMapResources(1, cuda_vbo_resource, 0);
		size_t num_bytes;
		cudaGraphicsResourceGetMappedPointer((void**)&ptr, &num_bytes, *cuda_vbo_resource);
		return ptr;
	}

	void unmapGLBufferObject(struct cudaGraphicsResource* cuda_vbo_resource) {
		cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
	}

	void copyArrayFromDevice(void* host, const void* device, struct cudaGraphicsResource** cuda_vbo_resource, int size) {

		if (cuda_vbo_resource && *cuda_vbo_resource) {
			device = mapGLBufferObject(cuda_vbo_resource);
		}

		cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost);

		if (cuda_vbo_resource && *cuda_vbo_resource) {
			unmapGLBufferObject(*cuda_vbo_resource);
		}
	}

	void setParameters(SimParams* hostParams) {
		cudaMemcpyToSymbol(cSimParams, hostParams, sizeof(SimParams));
	}

	uint iDivUp(uint a, uint b) {
		return (a % b != 0) ? (a / b + 1) : (a / b);
	}
	int divUpInt(int a, int b) {
		return (a + b - 1) / b;
	}

	void computeGridSize(uint n, uint blockSize, uint& numBlocks, uint& numThreads) {
		numThreads = min(blockSize, n);
		numBlocks = iDivUp(n, numThreads);
	}
	void integrateSystem(float* pos, float* vel, float* acc, float deltaTime, uint numParticles) {

		thrust::device_ptr<float4> d_pos4((float4*)pos);
		thrust::device_ptr<float4> d_vel4((float4*)vel);
		thrust::device_ptr<float4> d_acc4((float4*)acc);

		thrust::for_each(
			thrust::make_zip_iterator(thrust::make_tuple(d_pos4, d_vel4, d_acc4)),
			thrust::make_zip_iterator(thrust::make_tuple(d_pos4 + numParticles, d_vel4 + numParticles, d_acc4 + numParticles)),
			integrate_functor(deltaTime));
	}

	void forcesKernel(float* pos, float* acc, int numParticles) {
		uint numThreads = min((uint)256, numParticles);
		uint numBlocks = iDivUp(numParticles, numThreads);

		size_t smSz = numThreads * sizeof(float4);

		calculate_forces << <numBlocks, numThreads, smSz >> > ((float4*)pos, (float4*)acc, numParticles);
	}
	void calcHash(uint* gridParticleHash, uint* gridParticleIndex, float* pos, int numParticles) {

		uint numThreads, numBlocks;
		computeGridSize(numParticles, 256, numBlocks, numThreads);

		calcHashD << <numBlocks, numThreads >> > (gridParticleHash, gridParticleIndex, (float4*)pos, numParticles);
	}

	void reorderDataAndFindCellStart(uint* cellStart, uint* cellEnd, float* sortedPos, float* sortedVel, uint* gridParticleHash, uint* gridParticleIndex,
		float* oldPos, float* oldVel, uint numParticles, uint numCells) {

		uint numThreads, numBlocks;
		computeGridSize(numParticles, 256, numBlocks, numThreads);

		// set all cells to empty
		cudaMemset(cellStart, 0xffffffff, numCells * sizeof(uint));

#if USE_TEX
		cudaBindTexture(0, oldPosTex, oldPos, numParticles * sizeof(float4));
		cudaBindTexture(0, oldVelTex, oldVel, numParticles * sizeof(float4));
#endif

		uint smemSize = sizeof(uint) * (numThreads + 1);
		reorderDataAndFindCellStartD << <numBlocks, numThreads, smemSize >> > (
			cellStart, cellEnd,
			(float4*)sortedPos,
			(float4*)sortedVel,
			gridParticleHash,
			gridParticleIndex,
			(float4*)oldPos,
			(float4*)oldVel,
			numParticles
			);

#if USE_TEX
		cudaUnbindTexture(oldPosTex);
		cudaUnbindTexture(oldVelTex);
#endif
	}

	void collide(float* newVel, float* sortedPos, float* sortedVel, uint* gridParticleIndex,
		uint* cellStart, uint* cellEnd, uint numParticles, uint numCells) {

#if USE_TEX
		cudaBindTexture(0, oldPosTex, sortedPos, numParticles * sizeof(float4));
		cudaBindTexture(0, oldVelTex, sortedVel, numParticles * sizeof(float4));
		cudaBindTexture(0, cellStartTex, cellStart, numCells * sizeof(uint));
		cudaBindTexture(0, cellEndTex, cellEnd, numCells * sizeof(uint));
#endif

		uint numThreads, numBlocks;
		computeGridSize(numParticles, 64, numBlocks, numThreads);

		collideD << <numBlocks, numThreads >> > (
			(float4*)newVel,
			(float4*)sortedPos,
			(float4*)sortedVel,
			gridParticleIndex,
			cellStart,
			cellEnd,
			numParticles
			);

#if USE_TEX
		cudaUnbindTexture(oldPosTex);
		cudaUnbindTexture(oldVelTex);
		cudaUnbindTexture(cellStartTex);
		cudaUnbindTexture(cellEndTex);
#endif
	}

	void sortParticles(uint* dGridParticleHash, uint* dGridParticleIndex, uint numParticles) {
		thrust::sort_by_key(
			thrust::device_ptr<uint>(dGridParticleHash),
			thrust::device_ptr<uint>(dGridParticleHash + numParticles),
			thrust::device_ptr<uint>(dGridParticleIndex));
	}


}