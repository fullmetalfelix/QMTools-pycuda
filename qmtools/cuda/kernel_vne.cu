#include "kernel_common.h"

#define NATOMS PYCUDA_NATOMS
#define NX PYCUDA_NX
#define NY PYCUDA_NY
#define NZ PYCUDA_NZ
#define ADS PYCUDA_ADS
#define DIF PYCUDA_DIF



/// Performs a diffusion/generation/adsorption iteration of the nuclear pseudo-field.
__global__ void gpu_vne(
	int 	*types,
	float3 	*coords,
	float3 	grid0, 		// origin of the grid
	float 	dx,
	float 	*qube,
	float 	*qout,
	float 	*delta
	) {


	__shared__ float deltas[B_3];

	// shared atom types and pos
	__shared__ int styp[NATOMS];
	__shared__ float3 scrd[NATOMS];
	uint gidx = threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;
	if(gidx < NATOMS) {

		styp[gidx] = types[gidx];
		scrd[gidx] = coords[gidx];
		scrd[gidx].x -= grid0.x;
		scrd[gidx].y -= grid0.y;
		scrd[gidx].z -= grid0.z;
	}
	__syncthreads();
	// the atoms are now loaded!

	int3 r;
	uint ridx, widx;
	r.x = threadIdx.x + blockIdx.x*B;
	r.y = threadIdx.y + blockIdx.y*B;
	r.z = threadIdx.z + blockIdx.z*B;
	ridx = r.x + r.y*NX + r.z*NX*NY;



	float vnn = qube[ridx];
	float vnnout = 0;

	// *** COMPUTE DIFFUSION KERNEL ***************

	for(short x=-1; x<=1; x++) {
		for(short y=-1; y<=1; ++y) {
			for(short z=-1; z<=1; ++z) {

				float d = fabsf(x) + fabsf(y) + fabsf(z);
				if(d < 1) continue;

				int nnx = r.x+x;
				nnx = nnx + NX*((nnx<0)-(nnx >= NX));
				widx = nnx;

				nnx = r.y+y;
				nnx = nnx + NY*((nnx<0)-(nnx >= NY));
				widx += nnx*NX;

				nnx = r.z+z;
				nnx = nnx + NZ*((nnx<0)-(nnx >= NZ));
				widx += nnx*NX*NY;


				vnnout += DIF * qube[widx] / sqrtf(d);
			}
		}
	}
	vnnout += (1.0f - DIF * 16.794682450997072f) * vnn;

	// ********************************************
	// compute dissipation and nuclear generation *
	// apply adsorption
	vnnout -=  vnn * ADS;

	// compute voxel position
	float3 voxpos;
	voxpos.x = r.x * dx + 0.5f*dx;
	voxpos.y = r.y * dx + 0.5f*dx;
	voxpos.z = r.z * dx + 0.5f*dx;

	for(uint i=0; i<NATOMS; i++){

		float d = (voxpos.x-scrd[i].x)*(voxpos.x-scrd[i].x) + (voxpos.y-scrd[i].y)*(voxpos.y-scrd[i].y) + (voxpos.z-scrd[i].z)*(voxpos.z-scrd[i].z);
		d = sqrtf(d);
		if(d < sqrtf(3)*dx) d = styp[i] * exp(-d);
		else d = 0;

		vnnout += d;
	}
	// ********************************************

	if(r.x == 0 || r.x == NX-1 || r.y == 0 || r.y == NY-1 || r.z == 0 || r.z == NZ-1) vnnout = 0;

	qout[ridx] = vnnout;
	deltas[gidx] = fabsf(vnn - vnnout);
	__syncthreads();

	
	float psum = deltas[gidx];
	uint inext, vnext;

	// do a parallel reduction
	for(uint stride=1; stride < B_3; stride*=2) {

		inext = gidx + stride;
		vnext = inext * (inext < B_3);
		psum += deltas[vnext] * (inext < B_3);
		__syncthreads();

		deltas[gidx] = psum;
		__syncthreads();
	}
	// now the first thread has the total sum... maybe

	if(gidx == 0)
		atomicAdd(delta, psum);
}
