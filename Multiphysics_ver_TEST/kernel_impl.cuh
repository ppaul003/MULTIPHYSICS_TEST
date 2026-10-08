#ifndef _KERNEL_IMPL_CUH_
#define _KERNEL_IMPL_CUH_

#include <stdio.h>
#include <math.h>
#include <vector_types.h>
#include <helper_math.h>
//#include <math_constants.h>

#define EPS 0.01f
#define NUMSTEPS 20

// <CONSTANT PARAMS>
__constant__ ParticleSimParams cParticleParams;
__constant__ FieldGridParams cFieldGrid;
__constant__ FieldSolverParams cFieldSolver;

struct integrate_functor {

	float deltaTime;
	__host__ __device__ 
		integrate_functor(float delta_time) 
		: deltaTime(delta_time) {
	}

	template <typename Tuple>
	__device__ void operator()(Tuple t) {

		volatile float4 posData = thrust::get<0>(t);
		volatile float4 velData = thrust::get<1>(t);
		volatile float4 accData = thrust::get<2>(t);

		float3 pos = make_float3(posData.x, posData.y, posData.z);
		float3 vel = make_float3(velData.x, velData.y, velData.z);
		float3 acc = make_float3(accData.x, accData.y, accData.z);

		vel += acc * deltaTime;
		vel += cParticleParams.gravity * deltaTime;
		vel *= cParticleParams.globalDamping;

		// new position = old position + velocity * deltaTime
		pos += vel * deltaTime;

		// set this to zero to disable collisions with cube sides
#if 1
		if (pos.x > cParticleParams.boundary - velData.w) {
			pos.x = cParticleParams.boundary - velData.w;
			vel.x *= cParticleParams.boundaryDamping;
		}
		if (pos.x < -cParticleParams.boundary + velData.w) {
			pos.x = -cParticleParams.boundary + velData.w;
			vel.x *= cParticleParams.boundaryDamping;
		}
		if (pos.y > cParticleParams.boundary - velData.w) {
			pos.y = cParticleParams.boundary - velData.w;
			vel.y *= cParticleParams.boundaryDamping;
		}
		if (pos.z > cParticleParams.boundary - velData.w) {
			pos.z = cParticleParams.boundary - velData.w;
			vel.z *= cParticleParams.boundaryDamping;
		}
		if (pos.z < -cParticleParams.boundary + velData.w) {
			pos.z = -cParticleParams.boundary + velData.w;
			vel.z *= cParticleParams.boundaryDamping;
		}
		if (pos.y < -cParticleParams.boundary + velData.w) {
			pos.y = -cParticleParams.boundary + velData.w;
			vel.y *= cParticleParams.boundaryDamping;
		}

#endif
		// store new position and velocity
		thrust::get<0>(t) = make_float4(pos, posData.w);
		thrust::get<1>(t) = make_float4(vel, velData.w);
	}
};

#endif
