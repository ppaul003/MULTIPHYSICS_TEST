#ifndef KERNEL_H
#define KERNEL_H

#include <cstddef>
#include <cuda_runtime.h>
#include "paramsEM_kernel.cuh"

struct cudaGraphicsResource;

#ifdef __cplusplus
extern "C" {
#endif
    ///-----------------------------------------------------------------------------------------
    /// <CUDA HELPERS>
    ///-----------------------------------------------------------------------------------------
    void cudaInit(int argc, char** argv);
    void cudaGLInit(int argc, char** argv);

    void allocateArray(void** devPtr, size_t size);
    void freeArray(void* devPtr);
    void threadSync();

    void copyArrayToDevice(void* device, const void* host, int offset, int size);

    void registerGLBufferObject(unsigned int vbo, struct cudaGraphicsResource** cuda_vbo_resource);
    void unregisterGLBufferObject(struct cudaGraphicsResource* cuda_vbo_resource);
    void* mapGLBufferObject(struct cudaGraphicsResource** cuda_vbo_resource);
    void unmapGLBufferObject(struct cudaGraphicsResource* cuda_vbo_resource);
    void copyArrayFromDevice(void* host, const void* device,
        struct cudaGraphicsResource** cuda_vbo_resource, int size);
    ///-----------------------------------------------------------------------------------------
    /// </CUDA HELPERS>
    ///-----------------------------------------------------------------------------------------
    
    ///-----------------------------------------------------------------------------------------
    /// <PARTICLE SYSTEM CUDA SOLVER>
    ///-----------------------------------------------------------------------------------------
    void setParameters(SimParams* hostParams);
    void integrateSystem(float* pos, float* vel, float* acc, float deltaTime, unsigned int numParticles);
    void forcesKernel(float* pos, float* acc, int numParticles);
    void calcHash(unsigned int* gridParticleHash, unsigned int* gridParticleIndex, float* pos, int numParticles);

    void reorderDataAndFindCellStart(unsigned int* cellStart, unsigned int* cellEnd,
        float* sortedPos, float* sortedVel,
        unsigned int* gridParticleHash, unsigned int* gridParticleIndex,
        float* oldPos, float* oldVel,
        unsigned int numParticles, unsigned int numCells);

    void collide(float* newVel, float* sortedPos, float* sortedVel, unsigned int* gridParticleIndex,
        unsigned int* cellStart, unsigned int* cellEnd,
        unsigned int numParticles, unsigned int numCells);

    void sortParticles(unsigned int* dGridParticleHash, unsigned int* dGridParticleIndex, unsigned int numParticles);
    ///-----------------------------------------------------------------------------------------
    /// </PARTICLE SYSTEM CUDA SOLVER>
    ///-----------------------------------------------------------------------------------------

#ifdef __cplusplus
}
#endif
#endif