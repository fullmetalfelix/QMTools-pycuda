#include "kernel_common.h"

#define NATM PYCUDA_NATOMS

#define STEP PYCUDA_GRID_STEP
#define X0 PYCUDA_GRID_X0
#define Y0 PYCUDA_GRID_Y0
#define Z0 PYCUDA_GRID_Z0
#define NX PYCUDA_GRID_NX
#define NY PYCUDA_GRID_NY
#define NZ PYCUDA_GRID_NZ

#define SIGMA PYCUDA_SIGMA
#define CUTOFF_SQ PYCUDA_CUTOFF_SQ

// Add nuclear charge to electron density grid
__global__ void gpu_add_nuclear_charge(
	float* 		q,			// density grid
	int* 		types,		// atom types
	float3*		coords		// atom coordinates in BOHR
) {

	__shared__ int styp[100];
	__shared__ float3 scoords[100];

	// Copy atom coordinates and types to shared memory
	uint idx = threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;
	if(idx < NATM) {
		styp[idx] = types[idx];
		float3 tmp = coords[idx];
		scoords[idx].x = tmp.x;
		scoords[idx].y = tmp.y;
		scoords[idx].z = tmp.z;
	}

	__syncthreads();

	float3 pos;
	pos.x = X0 + ((blockIdx.x * B + threadIdx.x) + 0.5) * STEP;
	pos.y = Y0 + ((blockIdx.y * B + threadIdx.y) + 0.5) * STEP;
	pos.z = Z0 + ((blockIdx.z * B + threadIdx.z) + 0.5) * STEP;

    idx = (threadIdx.x + blockIdx.x*B);
	idx+= (threadIdx.y + blockIdx.y*B) * gridDim.x * B;
	idx+= (threadIdx.z + blockIdx.z*B) * gridDim.x * gridDim.y * B_2;

    q[idx] /= -STEP*STEP*STEP; // Convert to electron density in -e/bohr^3

	// Loop over atoms
	float c, r2;
	for (ushort i=0; i<NATM; i++) {

		// Squared distance from voxel to atom
		c = pos.x - scoords[i].x; r2 = c*c;
		c = pos.y - scoords[i].y; r2+= c*c;
		c = pos.z - scoords[i].z; r2+= c*c;

		// Add to voxel if within cutoff
        if (r2 <= CUTOFF_SQ) {
            q[idx] += styp[i] / (SIGMA*SIGMA*SIGMA*2*M_PI*sqrtf(2*M_PI)) * expf(-r2/(2*SIGMA*SIGMA));
        }

	}

}