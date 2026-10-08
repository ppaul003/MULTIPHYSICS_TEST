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
#include "fields_kernel.cuh"
#include "particles_kernel.cuh"

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

	void setParameters(ParticleSimParams* hostParams) {
		cudaMemcpyToSymbol(cParticleParams, hostParams, sizeof(ParticleSimParams));
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
// <FIELD SYSTEM HOST LAUNCHERS>
bool validFieldGrid(const FieldGridParams& g) {
    const unsigned limit = 2147483647u / (kFieldGlyphVerticesPerCell * sizeof(FieldGlyphVertex)); // legacy GL buffer-size query
    if (!g.dimensions.x || !g.dimensions.y || !g.dimensions.z ||
        g.dimensions.x > limit || g.dimensions.y > limit || g.dimensions.z > limit) return false;
    const uint64_t xy = uint64_t(g.dimensions.x) * g.dimensions.y;
    if (xy > limit || xy * g.dimensions.z > limit ||
        xy * g.dimensions.z != g.cellCount) return false;
    const float origins[] = {g.origin.x, g.origin.y, g.origin.z};
    const float sizes[] = {g.cellSize.x, g.cellSize.y, g.cellSize.z};
    const unsigned dims[] = {g.dimensions.x, g.dimensions.y, g.dimensions.z};
    for (int axis=0; axis<3; ++axis) {
        if (!std::isfinite(origins[axis]) || !std::isfinite(sizes[axis]) || sizes[axis] <= 0 ||
            !std::isfinite(origins[axis] + sizes[axis] * dims[axis])) return false;
    }
    return true;
}
bool setFieldGridParameters(const FieldGridParams* p) {
    return p && validFieldGrid(*p) &&
        cudaMemcpyToSymbol(cFieldGrid, p, sizeof(*p)) == cudaSuccess;
}
bool setFieldSolverParameters(const FieldSolverParams* p) {
    return p && p->boundaryMode == FieldBoundaryMode::GroundedDirichlet &&
        p->poissonIterations > 0 && std::isfinite(p->coulombSofteningM) &&
        p->coulombSofteningM >= 0 && std::isfinite(p->epsilon) && p->epsilon >= 0 &&
        cudaMemcpyToSymbol(cFieldSolver, p, sizeof(*p)) == cudaSuccess;
}
void freeFieldArrays(FieldBuffers& b) {
    float* scalars[] = {b.chargeDensity, b.electricPotential, b.electricPotentialScratch,
        b.temperature, b.electronDensity, b.electronTemperature};
    float4* vectors[] = {b.electricField, b.magneticField, b.currentDensity};
    for (auto p : scalars) if (p) cudaFree(p);
    for (auto p : vectors) if (p) cudaFree(p);
    b = FieldBuffers{};
}
bool allocateFieldArrays(FieldBuffers& b, unsigned count) {
    if (!count) return false;
    FieldBuffers next{};
    float** scalars[] = {&next.chargeDensity, &next.electricPotential, &next.electricPotentialScratch,
        &next.temperature, &next.electronDensity, &next.electronTemperature};
    float4** vectors[] = {&next.electricField, &next.magneticField, &next.currentDensity};
    for (auto p : scalars) if (cudaMalloc(reinterpret_cast<void**>(p), size_t(count)*sizeof(float)) != cudaSuccess) {
        freeFieldArrays(next); return false;
    }
    for (auto p : vectors) if (cudaMalloc(reinterpret_cast<void**>(p), size_t(count)*sizeof(float4)) != cudaSuccess) {
        freeFieldArrays(next); return false;
    }
    freeFieldArrays(b);
    b = next;
    return true;
}
bool copyFieldToDevice(void* device, const void* host, size_t bytes) {
    return bytes == 0 || (device && host &&
        cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice) == cudaSuccess);
}
bool copyFieldToHost(void* host, const void* device, size_t bytes) {
    return bytes == 0 || (device && host &&
        cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
}
bool clearScalarField(float* field, unsigned count) {
    if (!count) return true;
    if (!field) return false;
    clearScalarFieldD<<<(count+255)/256,256>>>(field, count);
    return cudaGetLastError() == cudaSuccess;
}
bool clearVectorField(float4* field, unsigned count) {
    if (!count) return true;
    if (!field) return false;
    clearVectorFieldD<<<(count+255)/256,256>>>(field, count);
    return cudaGetLastError() == cudaSuccess;
}
bool registerFieldBuffer(unsigned vbo, cudaGraphicsResource** resource) {
    return vbo && resource &&
        cudaGraphicsGLRegisterBuffer(resource, vbo, cudaGraphicsRegisterFlagsWriteDiscard) == cudaSuccess;
}
bool unregisterFieldBuffer(cudaGraphicsResource* resource) {
    return !resource || cudaGraphicsUnregisterResource(resource) == cudaSuccess;
}



bool solvePoissonJacobi(const float* rho, float* phiA, float* phiB,
    const FieldGridParams& grid, const FieldSolverParams& solver) {
    if (!rho || !phiA || !phiB || rho==phiA || rho==phiB || phiA==phiB ||
        grid.dimensions.x<3 || grid.dimensions.y<3 || grid.dimensions.z<3 ||
        !setFieldGridParameters(&grid) || !setFieldSolverParameters(&solver)) return false;
    if (!clearScalarField(phiA,grid.cellCount) || !clearScalarField(phiB,grid.cellCount)) return false;
    float* oldPhi=phiA;
    float* newPhi=phiB;
    for (unsigned step=0; step<solver.poissonIterations; ++step) {
        jacobiPoissonD<<<(grid.cellCount+255)/256,256>>>(rho,oldPhi,newPhi);
        if (cudaGetLastError()!=cudaSuccess) return false;
        float* swap=oldPhi; oldPhi=newPhi; newPhi=swap;
    }
    // Public potential pointer is stable for either odd or even iteration count.
    return oldPhi==phiA || cudaMemcpy(phiA,oldPhi,size_t(grid.cellCount)*sizeof(float),
        cudaMemcpyDeviceToDevice)==cudaSuccess;
}
bool computeFieldElectric(const float* phi, float4* electric, const FieldGridParams& grid) {
    if (!phi || !electric || !setFieldGridParameters(&grid)) return false;
    electricFieldFromPotentialD<<<(grid.cellCount+255)/256,256>>>(phi,electric);
    return cudaGetLastError()==cudaSuccess;
}


bool normalizeFieldWave(AnalyticWaveParams& wave) {
    if (!std::isfinite(wave.amplitudeVm) || !std::isfinite(wave.frequencyHz) ||
        wave.frequencyHz<0 || !std::isfinite(wave.phaseRad)) return false;
    if (!wave.enabled) return true;
    double d[3]={wave.propagationDirection.x,wave.propagationDirection.y,wave.propagationDirection.z};
    double p[3]={wave.polarization.x,wave.polarization.y,wave.polarization.z};
    double dn=0, pn=0;
    for (int a=0; a<3; ++a) {
        if (!std::isfinite(d[a]) || !std::isfinite(p[a])) return false;
        dn+=d[a]*d[a]; pn+=p[a]*p[a];
    }
    if (dn==0 || pn==0) return false;
    double dp=0;
    for (int a=0; a<3; ++a) { d[a]/=std::sqrt(dn); dp+=d[a]*p[a]; }
    double transverse=0;
    for (int a=0; a<3; ++a) { p[a]-=dp*d[a]; transverse+=p[a]*p[a]; }
    if (transverse<=1e-12*pn) return false;
    for (int a=0; a<3; ++a) p[a]/=std::sqrt(transverse);
    wave.propagationDirection=make_float3(float(d[0]),float(d[1]),float(d[2]));
    wave.polarization=make_float3(float(p[0]),float(p[1]),float(p[2]));
    return true;
}
bool addAnalyticWave(float4* electric, float4* magnetic, const FieldGridParams& grid,
    const AnalyticWaveParams& requested, double time) {
    auto wave=requested;
    if (!electric || !magnetic || !std::isfinite(time) ||
        !normalizeFieldWave(wave) || !setFieldGridParameters(&grid)) return false;
    if (!wave.enabled) return true;
    addAnalyticWaveD<<<(grid.cellCount+255)/256,256>>>(electric,magnetic,wave,time);
    return cudaGetLastError()==cudaSuccess;
}
bool computeFieldLorentz(const float4* positions, const float4* velocities,
    const ParticleFieldMarker* markers, float4* acceleration, unsigned count,
    const float4* electric, const float4* magnetic, const FieldGridParams& grid) {
    if (!count) return true;
    if (count>49152 || !positions || !velocities || !markers || !acceleration ||
        !electric || !magnetic || !setFieldGridParameters(&grid)) return false;
    lorentzAccelerationD<<<(count+255)/256,256>>>(positions,velocities,acceleration,
        markers,electric,magnetic,count);
    return cudaGetLastError()==cudaSuccess;
}


bool allocateFieldColors(float4** colors, unsigned count) {
    return colors && count &&
        cudaMalloc(reinterpret_cast<void**>(colors),size_t(count)*sizeof(float4))==cudaSuccess;
}
bool buildFieldRenderBuffers(const float4* electric, const float* scalar,
    const FieldGridParams& grid, const FieldRenderParams& s,
    cudaGraphicsResource* vectorResource, cudaGraphicsResource* scalarResource,
    const float4* vectorColors) {
    if (!electric || !scalar || !vectorResource || !scalarResource || vectorResource==scalarResource ||
        !s.stride || s.vectorScale>2 || !std::isfinite(s.vectorThreshold) || s.vectorThreshold<0 ||
        !std::isfinite(s.scalarThreshold) || s.scalarThreshold<0 ||
        !std::isfinite(s.vectorReference) || s.vectorReference<=0 ||
        !std::isfinite(s.scalarReference) || s.scalarReference<=0 ||
        !std::isfinite(s.logStrength) || s.logStrength<1 ||
        !std::isfinite(s.lengthInCells) || s.lengthInCells<=0 || !setFieldGridParameters(&grid)) return false;
    cudaGraphicsResource* resources[2]={vectorResource,scalarResource};
    if (cudaGraphicsMapResources(2,resources)!=cudaSuccess) return false;
    FieldGlyphVertex* lines=nullptr;
    FieldGlyphVertex* points=nullptr;
    size_t lineBytes=0, pointBytes=0;
    bool ok=cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&lines),&lineBytes,vectorResource)==cudaSuccess &&
        cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&points),&pointBytes,scalarResource)==cudaSuccess &&
        lineBytes>=size_t(grid.cellCount)*kFieldGlyphVerticesPerCell*sizeof(FieldGlyphVertex) &&
        pointBytes>=size_t(grid.cellCount)*sizeof(FieldGlyphVertex);
    if (ok) {
        buildElectricGlyphsD<<<(grid.cellCount+255)/256,256>>>(electric,lines,s,vectorColors);
        ok=cudaGetLastError()==cudaSuccess;
        if (ok) {
            buildScalarPointsD<<<(grid.cellCount+255)/256,256>>>(scalar,points,s);
            ok=cudaGetLastError()==cudaSuccess;
        }
    }
    const bool unmapped=cudaGraphicsUnmapResources(2,resources)==cudaSuccess;
    return ok && unmapped;
}

