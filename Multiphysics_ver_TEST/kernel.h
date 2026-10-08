#ifndef KERNEL_H
#define KERNEL_H

#include <cstddef>
#include <cuda_runtime.h>
#include "paramsEM_kernel.cuh"
#include "particleFieldType.h"

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
    void setParameters(ParticleSimParams* hostParams);
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

// <FIELD SYSTEM HOST INTERFACE>
// Single context thread, default stream. Each launcher uploads its owner's grid
// (and solver where needed), so alternating FieldSystem instances cannot use
// stale constant geometry. Arrays are device pointers unless marked host.
bool validFieldGrid(const FieldGridParams& grid);
bool setFieldGridParameters(const FieldGridParams* params);
bool setFieldSolverParameters(const FieldSolverParams* params);
bool allocateFieldArrays(FieldBuffers& buffers, unsigned count);
void freeFieldArrays(FieldBuffers& buffers);
bool copyFieldToDevice(void* device, const void* host, size_t bytes);
bool copyFieldToHost(void* host, const void* device, size_t bytes);
bool clearScalarField(float* field, unsigned count);
bool clearVectorField(float4* field, unsigned count);
bool solvePoissonJacobi(const float* rho, float* phiA, float* phiB,
    const FieldGridParams& grid, const FieldSolverParams& solver);
bool computeFieldElectric(const float* phi, float4* electric, const FieldGridParams& grid);
bool normalizeFieldWave(AnalyticWaveParams& wave);
bool addAnalyticWave(float4* electric, float4* magnetic,
    const FieldGridParams& grid, const AnalyticWaveParams& wave, double time);
bool computeFieldLorentz(const float4* positions, const float4* velocities,
    const ParticleFieldMarker* markers, float4* acceleration, unsigned count,
    const float4* electric, const float4* magnetic, const FieldGridParams& grid);
bool registerFieldBuffer(unsigned vbo, cudaGraphicsResource** resource);
bool unregisterFieldBuffer(cudaGraphicsResource* resource);
bool allocateFieldColors(float4** colors, unsigned count);
bool buildFieldRenderBuffers(const float4* electric, const float* scalar,
    const FieldGridParams& grid, const FieldRenderParams& settings,
    cudaGraphicsResource* vectorResource, cudaGraphicsResource* scalarResource,
    const float4* vectorColors = nullptr);
#endif
#endif
