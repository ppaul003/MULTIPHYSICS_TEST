#ifndef _KERNEL_IMPL_CUH_
#define _KERNEL_IMPL_CUH_
#include <math.h>
#include <helper_math.h>
#include <thrust/tuple.h>
#include "paramsEM_kernel.cuh"
#include "particleFieldType.h"

__constant__ ParticleSimParams cParticleParams{};
__constant__ FieldGridParams cFieldGrid{};
__constant__ FieldSolverParams cFieldSolver;
constexpr double kVacuumEpsilon0 = 8.8541878128e-12;
constexpr double kCoulombSI = 1.0 / (12.56637061435917295385 * kVacuumEpsilon0);

__device__ __forceinline__ bool finite3(float3 v) {
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}
__device__ __forceinline__ float3 scaled3(float3 v, double s) {
    return make_float3(float(double(v.x)*s),float(double(v.y)*s),float(double(v.z)*s));
}
__device__ __forceinline__ void addAcceleration(float4* a, unsigned i, float3 v) {
    a[i]=make_float4(make_float3(a[i])+v,0);
}
__global__ void initializeAccelerationD(float4* a, const ParticleFieldMarker* markers, unsigned n) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) a[i]=make_float4(markers[i].kind==ParticleKind::Inactive ? make_float3(0) : cParticleParams.gravity,0);
}
__device__ __forceinline__ void constrainAxis(float& p,float& v,float lo,float hi) {
    if(p>hi) {p=hi; if(v>0) v*=cParticleParams.boundaryDamping;}
    else if(p<lo) {p=lo; if(v<0) v*=cParticleParams.boundaryDamping;}
}
struct integrate_functor {
    float deltaTime;
    __host__ __device__ explicit integrate_functor(float dt):deltaTime(dt) {}
    template<class Tuple> __device__ void operator()(Tuple t) const {
        const ParticleFieldMarker marker=thrust::get<3>(t);
        if(marker.kind==ParticleKind::Inactive) return;
        const float4 pd=thrust::get<0>(t), vd=thrust::get<1>(t), ad=thrust::get<2>(t);
        float3 p=make_float3(pd),v=make_float3(vd);
        if(!finite3(p)||!finite3(v)||!finite3(make_float3(ad))||
           !isfinite(vd.w)||vd.w<0||vd.w>cParticleParams.boundary) return;
        // All contributors are SI accelerations; apply dt exactly once.
        v+=scaled3(make_float3(ad),double(deltaTime)/cParticleParams.metersPerWorldUnit);
        v*=cParticleParams.globalDamping;
        p+=v*deltaTime;
        if(!finite3(p)||!finite3(v)) return;
        const float lo=-cParticleParams.boundary+vd.w, hi=cParticleParams.boundary-vd.w;
        constrainAxis(p.x,v.x,lo,hi); constrainAxis(p.y,v.y,lo,hi); constrainAxis(p.z,v.z,lo,hi);
        thrust::get<0>(t)=make_float4(p,pd.w);
        thrust::get<1>(t)=make_float4(v,vd.w);
    }
};
#endif
