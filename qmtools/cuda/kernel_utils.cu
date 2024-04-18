#include "kernel_common.h"


/// Populate a field with an initial seed for the electron density.
__global__ void gpu_density_seed(
	float 	gridStep, 	// grid step size
	float3 	grid0,
	dim3 	n,
	int 	*Zs,
	float3 	*coords,
	float 	*qfield 	// output density field
	) {

	__shared__ float3 atom;
	__shared__ int Z;


	uint gidx = threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;

	// get the atom coords from global memory
	if(gidx == 0) {
		atom = coords[blockIdx.x];
		atom.x -= grid0.x;
		atom.y -= grid0.y;
		atom.z -= grid0.z;
		Z = Zs[blockIdx.x];
	}
	__syncthreads();

	// compute in which block the atom is located
	// it might be at the edge between two blocks

	// this is the index in the unwrapped grid where the thread
	// should place its share of the nuclear charge
	uint3 i0;
	i0.x = (uint)floorf((atom.x) / gridStep) + threadIdx.x;
	i0.y = (uint)floorf((atom.y) / gridStep) + threadIdx.y;
	i0.z = (uint)floorf((atom.z) / gridStep) + threadIdx.z;


	// calculate the charge factor
	float p = (float)Z;
	p *= fabsf(atom.x - i0.x * gridStep) / gridStep;
	p *= fabsf(atom.y - i0.y * gridStep) / gridStep;
	p *= fabsf(atom.z - i0.z * gridStep) / gridStep;

	// save the electronic charge
	gidx = i0.x + i0.y*n.x + i0.z*n.x*n.y;
	qfield[gidx] = p;
}


/// Rescale the values in a qube.
__global__ void gpu_rescale(float *qube, float factor) {

	uint3 r;
	uint ridx;

	r.x = threadIdx.x + blockIdx.x*B;
	r.y = threadIdx.y + blockIdx.y*B;
	r.z = threadIdx.z + blockIdx.z*B;
	ridx = r.x + r.y*gridDim.x*B + r.z*gridDim.x*gridDim.y*B_2;

	qube[ridx] = qube[ridx] * factor;
}

/// computes the sum of all elements in a grid
__global__ void gpu_total(float *qube, float* qtot) {

	__shared__ float q[B_3];
	uint3 r;
	uint ridx, widx;

	r.x = threadIdx.x + blockIdx.x*B;
	r.y = threadIdx.y + blockIdx.y*B;
	r.z = threadIdx.z + blockIdx.z*B;
	ridx = r.x + r.y*gridDim.x*B + r.z*gridDim.x*gridDim.y*B_2;
	widx = threadIdx.x + threadIdx.y*B + threadIdx.z*B_2;
	
	q[widx] = qube[ridx];
	__syncthreads();

	// do the partial sums: each thread adds the next
	float psum = q[widx];
	for(ushort stride=1; stride<B_3; stride*=2) {

		ushort idxnext = widx + stride;
		idxnext *= (idxnext < B_3);

		psum += q[idxnext];
		__syncthreads();

		q[widx] = psum;
		__syncthreads();
	}

	if(widx == 0) atomicAdd(qtot, psum);
}

/// Compare two density grids.
__global__ void gpu_qdiff(float *qube, float *qref, float *qdiff) {

	__shared__ float q[B_3];

	uint3 r;
	uint ridx, widx;

	r.x = threadIdx.x + blockIdx.x*B;
	r.y = threadIdx.y + blockIdx.y*B;
	r.z = threadIdx.z + blockIdx.z*B;
	ridx = r.x + r.y*gridDim.x*B + r.z*gridDim.x*gridDim.y*B_2;
	widx = threadIdx.x + threadIdx.y*B + threadIdx.z*B_2;
	
	q[widx] = fabsf(qube[ridx] - qref[ridx]);
	__syncthreads();

	// do the partial sums: each thread adds the next
	float psum = q[widx];
	for(ushort stride=1; stride<B_3; stride*=2) {

		ushort idxnext = widx + stride;
		idxnext *= (idxnext < B_3);

		psum += q[idxnext];
		__syncthreads();

		q[widx] = psum;
		__syncthreads();
	}
	
	//psum /= B_3; // average difference in this block
	if(widx == 0) atomicAdd(qdiff, psum);
}

/// Compare two density grids.
__global__ void gpu_qdiff_rel(float *qube, float *qref, float *qdiff) {

	__shared__ float q[B_3];

	uint3 r;
	uint ridx, widx;

	r.x = threadIdx.x + blockIdx.x*B;
	r.y = threadIdx.y + blockIdx.y*B;
	r.z = threadIdx.z + blockIdx.z*B;
	ridx = r.x + r.y*gridDim.x*B + r.z*gridDim.x*gridDim.y*B_2;
	widx = threadIdx.x + threadIdx.y*B + threadIdx.z*B_2;
	
	q[widx] = fabsf(qube[ridx] - qref[ridx]) / qref[ridx];
	__syncthreads();

	// do the partial sums: each thread adds the next
	float psum = q[widx];
	for(ushort stride=1; stride<B_3; stride*=2) {

		ushort idxnext = widx + stride;
		idxnext *= (idxnext < B_3);

		psum += q[idxnext];
		__syncthreads();

		q[widx] = psum;
		__syncthreads();
	}
	
	//psum /= B_3; // average difference in this block
	if(widx == 0) atomicAdd(qdiff, psum);
}