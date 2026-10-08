#ifndef _FIELDS_KERNEL_CUH_
#define _FIELDS_KERNEL_CUH_

#include "paramsEM_kernel.cuh"
#include "particleFieldType.h"
#include "kernel_impl.cuh"

#include <stdio.h>
#include <math.h>
#include <vector_types.h>
#include <helper_math.h>

#define EPSILON 8.8541878128e-12f
#define K1 6.2831853071795864769f
#define K2 299792458.0f
///////////////////////////////////////
__global__
void clearScalarFieldD(
	float* field,
	unsigned count
);
__global__
void clearVectorFieldD(
	float4* field,
	unsigned count
);
__global__
void jacobiPoissonD(
	const float* rho,
	const float* oldPhi,
	float* newPhi
);
__global__
void electricFieldFromPotentialD(
	const float* phi,
	float4* electric
);
__global__
void addAnalyticWaveD(
	float4* electric,
	float4* magnetic,
	AnalyticWaveParams wave,
	double time
);
__global__
void lorentzAccelerationD(
	const float4* positions,
	const float4* velocities,
	float4* acceleration,
	const ParticleFieldMarker* markers,
	const float4* electric,
	const float4* magnetic,
	unsigned count
);
__global__
void buildElectricGlyphsD(
	const float4* electric,
	FieldGlyphVertex* vertices,
	FieldRenderParams settings,
	const float4* colors
);
__global__
void buildScalarPointsD(
	const float* scalar,
	FieldGlyphVertex* vertices,
	FieldRenderParams settings
);
///////////////////////////////
///////////////////////////////
__device__ float3 fieldCellCenter(unsigned i);
__device__ double fieldVectorMagnitude(float3 v);
///////////////////////////////
__device__
uint fieldIndex(
	uint x, 
	uint y, 
	uint z, 
	uint3 dim
);
__device__
float fieldDerivative(
	const float* phi,
	unsigned i,
	unsigned coordinate,
	unsigned dimension,
	unsigned stride,
	float spacing
);
__device__
float3 sampleFieldVector(
	const float4* field,
	float3 p
);
__device__
FieldGlyphVertex fieldVertex(
	float3 p,
	float4 color
);
__device__
bool fieldGlyphSample(
	unsigned i,
	unsigned stride
);
//////////////////////////
// <FIELD SYSTEM KERNELS>
__global__
void clearScalarFieldD(
	float* field,
	unsigned count) {

	const uint i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < count)
		field[i] = 0.0f;
}
__global__
void clearVectorFieldD(
	float4* field,
	unsigned count) {

	const uint i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < count)
		field[i] = make_float4(0, 0, 0, 0);
}
// Cell-centered finite-volume Jacobi: grounded physical faces, vacuum SI units.
__global__ void jacobiPoissonD(const float* rho,const float* oldPhi,float* newPhi) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=cFieldGrid.cellCount) return;
    const uint3 d=cFieldGrid.dimensions;
    const unsigned coord[3]={i%d.x,(i/d.x)%d.y,i/(d.x*d.y)};
    const unsigned dims[3]={d.x,d.y,d.z}, stride[3]={1,d.x,d.x*d.y};
    const double h[3]={cFieldGrid.cellSize.x*cFieldGrid.metersPerWorldUnit,
        cFieldGrid.cellSize.y*cFieldGrid.metersPerWorldUnit,cFieldGrid.cellSize.z*cFieldGrid.metersPerWorldUnit};
    double numerator=double(rho[i])/kVacuumEpsilon0,denominator=0;
    for(int a=0;a<3;++a) {
        const double w=1.0/(h[a]*h[a]);
        if(coord[a]>0) {numerator+=w*oldPhi[i-stride[a]];denominator+=w;} else denominator+=2*w;
        if(coord[a]+1<dims[a]) {numerator+=w*oldPhi[i+stride[a]];denominator+=w;} else denominator+=2*w;
    }
    newPhi[i]=float(numerator/denominator);
}
__device__ float groundedDerivative(const float* phi,unsigned i,unsigned c,unsigned n,unsigned stride,float h) {
    if(n==1) return 0;
    if(c==0) return float((double(phi[i])+double(phi[i+stride])/3.0)/h);
    if(c+1==n) return float(-(double(phi[i])+double(phi[i-stride])/3.0)/h);
    return float((double(phi[i+stride])-phi[i-stride])/(2.0*h));
}
__global__
void electricFieldFromPotentialD(
	const float* phi,
	float4* electric, bool groundedFaces) {

	const uint i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= cFieldGrid.cellCount)
		return;

	const uint3 d = cFieldGrid.dimensions;
	const uint x = i % d.x;
	const uint y = (i / d.x) % d.y;
	const uint z = i / (d.x * d.y);
	const float3 h = cFieldGrid.cellSize;

	electric[i] = make_float4(
		float(-double(groundedFaces ? groundedDerivative(phi, i, x, d.x, 1, h.x) : fieldDerivative(phi, i, x, d.x, 1, h.x))/cFieldGrid.metersPerWorldUnit),
		float(-double(groundedFaces ? groundedDerivative(phi, i, y, d.y, d.x, h.y) : fieldDerivative(phi, i, y, d.y, d.x, h.y))/cFieldGrid.metersPerWorldUnit),
		float(-double(groundedFaces ? groundedDerivative(phi, i, z, d.z, d.x * d.y, h.z) : fieldDerivative(phi, i, z, d.z, d.x * d.y, h.z))/cFieldGrid.metersPerWorldUnit),
		0.0f
	);
}
// Propagation and transverse polarization are normalized by the host wrapper.
__global__
void addAnalyticWaveD(
	float4* electric,
	float4* magnetic,
	AnalyticWaveParams wave,
	double time) {

	const uint i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= cFieldGrid.cellCount)
		return;

	const float3 r = fieldCellCenter(i);
	const float3 k = wave.propagationDirection;

	const double distance = double(k.x) * r.x +
		double(k.y) * r.y + double(k.z) * r.z;

	const double phase = K1 * double(wave.frequencyHz) *
		(distance * cFieldGrid.metersPerWorldUnit / K2 - time) + wave.phaseRad;

	const float3 e = wave.polarization *
		static_cast<float>(wave.amplitudeVm * cos(phase));

	const float3 b = cross(k, e) / K2;

	electric[i] = make_float4(make_float3(electric[i]) + e, 0);
	magnetic[i] = make_float4(make_float3(magnetic[i]) + b, 0);
}
__global__
void lorentzAccelerationD(
	const float4* positions,
	const float4* velocities,
	float4* acceleration,
	const ParticleFieldMarker* markers,
	const float4* electric,
	const float4* magnetic,
	unsigned count) {

	const uint i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= count)
		return;

	const auto marker = markers[i];


	if (marker.kind == ParticleKind::Inactive ||
		marker.chargeC == 0 || marker.chargeToMass == 0)
		return;

	const float3 p = make_float3(positions[i]);
	const float3 v = scaled3(make_float3(velocities[i]), cFieldGrid.metersPerWorldUnit);

	const float3 e = sampleFieldVector(electric, p);
	const float3 b = sampleFieldVector(magnetic, p);

	// SI acceleration is additive; sampling happens once for the composed E/B.
	addAcceleration(acceleration, i, (e + cross(v, b)) * marker.chargeToMass);
}
__global__
void buildElectricGlyphsD(
	const float4* electric,
	FieldGlyphVertex* vertices,
	FieldRenderParams settings,
	const float4* colors) {

	const uint i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= cFieldGrid.cellCount)
		return;

	const uint base = i * kFieldGlyphVerticesPerCell;

	for (uint v = 0; v < kFieldGlyphVerticesPerCell; v++) {

		vertices[base + v] =
			fieldVertex(
				make_float3(0),
				make_float4(0, 0, 0, 0)
			);
	}

	if (!fieldGlyphSample(i, settings.stride))
		return;

	const float3 value = make_float3(electric[i]);
	const double magnitude = fieldVectorMagnitude(value);

	if (!isfinite(magnitude) ||
		magnitude <= 0 ||
		magnitude < settings.vectorThreshold)
		return;

	double fraction = 1.0;

	if (settings.vectorScale != 0) {

		fraction = fmin(
			1.0,
			magnitude / settings.vectorReference
		);

		if (settings.vectorScale == 2) {

			fraction =
				log1p(settings.logStrength * fraction) /
				log1p(settings.logStrength);
		}
	}

	const float h = fminf(
		cFieldGrid.cellSize.x,
		fminf(cFieldGrid.cellSize.y, cFieldGrid.cellSize.z)
	);

	const float length =
		settings.lengthInCells * h * static_cast<float>(fraction);

	if (length <= 0)
		return;

	const float3 direction = make_float3(
		float(value.x / magnitude),
		float(value.y / magnitude),
		float(value.z / magnitude)
	);

	const float3 origin = fieldCellCenter(i);
	const float3 tip = origin + direction * length;

	const float3 reference =
		fabsf(direction.y) < 0.9f
		? make_float3(0, 1, 0)
		: make_float3(1, 0, 0);

	const float3 side = normalize(cross(direction, reference));

	const float3 neck = tip - direction * (length * 0.25f);
	const float width = length * 0.12f;

	float4 color =
		colors
		? colors[i]
		: settings.vectorColor;
	// Visual opacity only. The arrowhead lies in one plane in world space.
	color.w = fminf(0.8f, fmaxf(0.0f, color.w)) * (0.6f + 0.4f * static_cast<float>(fraction));

	vertices[base] = fieldVertex(origin, color);
	vertices[base + 1] = fieldVertex(tip, color);
	vertices[base + 2] = fieldVertex(tip, color);
	vertices[base + 3] = fieldVertex(neck + side * width, color);
	vertices[base + 4] = fieldVertex(tip, color);
	vertices[base + 5] = fieldVertex(neck - side * width, color);
}
__global__
void buildScalarPointsD(
	const float* scalar,
	FieldGlyphVertex* vertices,
	FieldRenderParams settings) {

	const uint i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= cFieldGrid.cellCount)
		return;

	vertices[i] = fieldVertex(
		make_float3(0),
		make_float4(0, 0, 0, 0)
	);

	if (!fieldGlyphSample(i, settings.stride))
		return;

	const double value = scalar[i];
	const double magnitude = fabs(value);

	if (!isfinite(value) ||
		magnitude <= 0 ||
		magnitude < settings.scalarThreshold)
		return;

	const float relative = static_cast<float>(sqrt(fmin(1.0, magnitude / settings.scalarReference)));
	const float brightness = 0.35f + 0.65f * relative;

	const float3 color =
		value > 0
		? make_float3(1, 0.45f, 0.08f)
		: make_float3(0.15f, 0.55f, 1);

	vertices[i] = fieldVertex(
		fieldCellCenter(i),
		make_float4(color * brightness, 0.4f + 0.4f * relative)
	);
}
///-----------------------------------------------------------------------------------------
/// <FIELD SYSTEM DEVICE FUNCTIONS>
///-----------------------------------------------------------------------------------------
__device__
uint fieldIndex(
	uint x, 
	uint y, 
	uint z, 
	uint3 dim) {
	return x +
		y * dim.x +
		z * dim.x * dim.y;
}
__device__
float fieldDerivative(
	const float* phi,
	unsigned i,
	unsigned coordinate,
	unsigned dimension,
	unsigned stride,
	float spacing) {
	if (dimension == 1)
		return 0.0f;
	const uint lo =
		coordinate == 0
		? i
		: i - stride;
	const uint hi =
		coordinate + 1 == dimension
		? i
		: i + stride;
	const float width =
		(coordinate == 0 || coordinate + 1 == dimension)
		? spacing
		: 2.0f * spacing;
	return (phi[hi] - phi[lo]) / width;
}
__device__
float3 fieldCellCenter(unsigned i) {
	const uint3 d = cFieldGrid.dimensions;
	return
		cFieldGrid.origin +
		make_float3(
			(i % d.x) + 0.5f,
			((i / d.x) % d.y) + 0.5f,
			(i / (d.x * d.y)) + 0.5f
		) * cFieldGrid.cellSize;
}
__device__
float3 sampleFieldVector(
	const float4* field,
	float3 p) {
	if (!field || !finite3(p)) return make_float3(nanf(""));
	const uint3 d = cFieldGrid.dimensions;
	const float3 coordinate =
		(p - cFieldGrid.origin) / cFieldGrid.cellSize -
		make_float3(0.5f);
	const float3 u = make_float3(
		fminf(fmaxf(coordinate.x, 0), float(d.x - 1)),
		fminf(fmaxf(coordinate.y, 0), float(d.y - 1)),
		fminf(fmaxf(coordinate.z, 0), float(d.z - 1))
	);
	const uint3 lo = make_uint3(
		unsigned(floorf(u.x)),
		unsigned(floorf(u.y)),
		unsigned(floorf(u.z))
	);
	const uint3 hi = make_uint3(
		min(lo.x + 1, d.x - 1),
		min(lo.y + 1, d.y - 1),
		min(lo.z + 1, d.z - 1)
	);
	const float3 f = u - make_float3(
		float(lo.x),
		float(lo.y),
		float(lo.z)
	);
	float3 sum = make_float3(0);
	for (unsigned z = 0; z < 2; z++) {
		for (unsigned y = 0; y < 2; y++) {
			for (unsigned x = 0; x < 2; x++) {
				const float w =
					(x ? f.x : 1 - f.x) *
					(y ? f.y : 1 - f.y) *
					(z ? f.z : 1 - f.z);
				sum += make_float3(field[fieldIndex(
					x ? hi.x : lo.x,
					y ? hi.y : lo.y,
					z ? hi.z : lo.z,
					d)]
				) * w;
			}
		}
	}
	return sum;
}
__device__
double fieldVectorMagnitude(float3 v) {
	return sqrt(
		double(v.x) * v.x +
		double(v.y) * v.y +
		double(v.z) * v.z
	);
}
__device__
FieldGlyphVertex fieldVertex(
	float3 p,
	float4 color) {
	FieldGlyphVertex v;
	v.position = make_float4(p, 1);
	v.color = color;
	return v;
}
__device__
bool fieldGlyphSample(
	unsigned i,
	unsigned stride) {
	const uint3 d = cFieldGrid.dimensions;
	return (i % d.x) % stride == 0 &&
		((i / d.x) % d.y) % stride == 0 &&
		(i / (d.x * d.y)) % stride == 0;
}
///-----------------------------------------------------------------------------------------
/// </FIELD SYSTEM DEVICE FUNCTIONS>
///-----------------------------------------------------------------------------------------

// Same cell-centered clamped CIC convention as sampleFieldVector.
__global__ void depositChargeCurrentD(const float4* pos,const float4* vel,
    const ParticleFieldMarker* markers,float* rho,float4* current,unsigned count) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count||markers[i].kind==ParticleKind::Inactive||markers[i].chargeC==0) return;
    if(!finite3(make_float3(pos[i]))||!finite3(make_float3(vel[i]))||!isfinite(markers[i].chargeC)) return;
    const uint3 d=cFieldGrid.dimensions;
    const float3 coordinate=(make_float3(pos[i])-cFieldGrid.origin)/cFieldGrid.cellSize-make_float3(0.5f);
    const float3 u=make_float3(fminf(fmaxf(coordinate.x,0),float(d.x-1)),
        fminf(fmaxf(coordinate.y,0),float(d.y-1)),fminf(fmaxf(coordinate.z,0),float(d.z-1)));
    const uint3 lo=make_uint3(unsigned(floorf(u.x)),unsigned(floorf(u.y)),unsigned(floorf(u.z)));
    const uint3 hi=make_uint3(min(lo.x+1,d.x-1),min(lo.y+1,d.y-1),min(lo.z+1,d.z-1));
    const float3 f=u-make_float3(float(lo.x),float(lo.y),float(lo.z));
    const double scale=cFieldGrid.metersPerWorldUnit;
    const double volume=(double(cFieldGrid.cellSize.x)*scale)*(double(cFieldGrid.cellSize.y)*scale)*(double(cFieldGrid.cellSize.z)*scale);
    for(unsigned z=0;z<2;++z) for(unsigned y=0;y<2;++y) for(unsigned x=0;x<2;++x) {
        const double w=(x?f.x:1-f.x)*(y?f.y:1-f.y)*(z?f.z:1-f.z);
        if(w==0) continue;
        const unsigned cell=fieldIndex(x?hi.x:lo.x,y?hi.y:lo.y,z?hi.z:lo.z,d);
        const double density=w*double(markers[i].chargeC)/volume;
        atomicAdd(rho+cell,float(density));
        atomicAdd(&current[cell].x,float(density*double(vel[i].x)*scale));
        atomicAdd(&current[cell].y,float(density*double(vel[i].y)*scale));
        atomicAdd(&current[cell].z,float(density*double(vel[i].z)*scale));
    }
}
__global__ void addUniformFieldD(float4* e,float4* b,UniformEMField source) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=cFieldGrid.cellCount||!source.enabled) return;
    e[i]=make_float4(make_float3(e[i])+source.electricVm,0);
    b[i]=make_float4(make_float3(b[i])+source.magneticT,0);
}
#endif
