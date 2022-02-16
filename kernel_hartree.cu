#include "kernel_common.h"

#define NATM PYCUDA_NATOMS

#define STEP PYCUDA_GRID_STEP
#define X0 PYCUDA_GRID_X0
#define Y0 PYCUDA_GRID_Y0
#define Z0 PYCUDA_GRID_Z0
#define NX PYCUDA_GRID_NX
#define NY PYCUDA_GRID_NY
#define NZ PYCUDA_GRID_NZ

/*
#define VSTEP PYCUDA_VGRID_STEP
#define VX0 PYCUDA_VGRID_X0
#define VY0 PYCUDA_VGRID_Y0
#define VZ0 PYCUDA_VGRID_Z0
*/


#define Bp 9
#define Bp1 10
#define Bp1_2 100
#define Bp1_3 1000



// compute an initial guess
__global__ void __launch_bounds__(512, 4) gpu_hartree_guess(
	int* 		types,
	float3*		coords, 	// atom coordinates in BOHR
	float*		V 			// output hartree qube
	){

	__shared__ int styp[100];
	__shared__ float3 scoords[100];

	uint sidx = threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;

	// load atoms in shmem
	if(sidx < NATM) {
		styp[sidx] = types[sidx];
		scoords[sidx] = coords[sidx];
	}
	__syncthreads();

	float hartree = 0;
	float c,r;

	// compute voxel position in the output V grid
	float3 V_pos;
	V_pos.x = X0 + (blockIdx.x * B + threadIdx.x) * STEP + 0.5f*STEP;
	V_pos.y = Y0 + (blockIdx.y * B + threadIdx.y) * STEP + 0.5f*STEP;
	V_pos.z = Z0 + (blockIdx.z * B + threadIdx.z) * STEP + 0.5f*STEP;


	// add the nuclear potential
	for(ushort i=0; i<NATM; i++) {

		c = V_pos.x - scoords[i].x; r = c*c;
		c = V_pos.y - scoords[i].y; r+= c*c;
		c = V_pos.z - scoords[i].z; r+= c*c;
		r = sqrtf(r);
		hartree += styp[i] / r;
	}
	
	// write results
	sidx = (threadIdx.x + blockIdx.x*B);
	sidx+= (threadIdx.y + blockIdx.y*B) * NX;
	sidx+= (threadIdx.z + blockIdx.z*B) * NX*NY;
	V[sidx] = -hartree;
}




// do one jacobi iteration of the posson solver
__global__ void __launch_bounds__(512, 4) gpu_hartree_iteration(
	float* 		Q,			// density grid
	float*		V, 			// current V guess
	float*		Vout 		// updated V
	){

	__shared__ float shQ[Bp1_3];

	// global memory address
	uint gidx = threadIdx.x + blockIdx.x*B + (threadIdx.y+blockIdx.y*B)*NX + (threadIdx.z+blockIdx.z*B)*NX*NY;
	

	// data destination address in shared memory - for main block of data
	int sidx = (threadIdx.x+1) + (threadIdx.y+1) * Bp1 + (threadIdx.z+1) * Bp1_2;
	
	// load the main block of data in shared mem
	shQ[sidx] = V[gidx];
	__syncthreads();

	short flag = -1;
	if(threadIdx.x == 0){ // load last x of previous block

		sidx = (0) + (threadIdx.y+1) * Bp1 + (threadIdx.z+1) * Bp1_2;
		gidx = (B-1) + (blockIdx.x-1)*B + (threadIdx.y+blockIdx.y*B)*NX + (threadIdx.z+blockIdx.z*B)*NX*NY;
		flag = (blockIdx.x != 0);

	}else if(threadIdx.x == 1){ // first x of next block

		sidx = (Bp) + (threadIdx.y+1) * Bp1 + (threadIdx.z+1) * Bp1_2;
		gidx = (0) + (blockIdx.x+1)*B + (threadIdx.y+blockIdx.y*B)*NX + (threadIdx.z+blockIdx.z*B)*NX*NY;
		flag = (blockIdx.x+1 < gridDim.x);

	}else if(threadIdx.x == 2){ // load last y of blocky-1

		sidx = (threadIdx.y+1) + 0 * Bp1 + (threadIdx.z+1) * Bp1_2;
		gidx = threadIdx.y + blockIdx.x*B + (B-1+(blockIdx.y-1)*B)*NX + (threadIdx.z+blockIdx.z*B)*NX*NY;
		flag = (blockIdx.y != 0);

	}else if(threadIdx.x == 3){ // load first y of blocky+1

		sidx = (threadIdx.y+1) + Bp * Bp1 + (threadIdx.z+1) * Bp1_2;
		gidx = threadIdx.y + blockIdx.x*B + (0+(blockIdx.y+1)*B)*NX + (threadIdx.z+blockIdx.z*B)*NX*NY;
		flag = (blockIdx.y+1 < gridDim.y);

	}else if(threadIdx.x == 4){ // load last z of blockz-1

		sidx = (threadIdx.y+1) + (threadIdx.z+1) * Bp1 + 0 * Bp1_2;
		gidx = threadIdx.y + blockIdx.x*B + (threadIdx.z + blockIdx.y*B)*NX + (B-1+(blockIdx.z-1)*B)*NX*NY;
		flag = (blockIdx.z != 0);

	}else if(threadIdx.x == 5){ // load first z of blockyz+1

		sidx = (threadIdx.y+1) + (threadIdx.z+1) * Bp1 + Bp * Bp1_2;
		gidx = threadIdx.y + blockIdx.x*B + (threadIdx.z + blockIdx.y*B)*NX + (0+(blockIdx.z+1)*B)*NX*NY;
		flag = (blockIdx.z+1 < gridDim.z);

	}

	if(threadIdx.x < 6){

		if(flag>0) shQ[sidx] = V[gidx];
		else shQ[sidx] = 0;
	}
	__syncthreads();


	// restore the shmem address of this thread
	sidx = (threadIdx.x+1) + (threadIdx.y+1) * Bp1 + (threadIdx.z+1) * Bp1_2;
	gidx = threadIdx.x + blockIdx.x*B + (threadIdx.y+blockIdx.y*B)*NX + (threadIdx.z+blockIdx.z*B)*NX*NY;


	float vnew = shQ[sidx-1]+shQ[sidx+1] + shQ[sidx+Bp1]+shQ[sidx-Bp1] + shQ[sidx+Bp1_2]+shQ[sidx-Bp1_2];
	vnew -= Q[gidx]/STEP;
	vnew /= 6.0f;
	

	if((threadIdx.x+blockIdx.x*B == 0) || (threadIdx.x+blockIdx.x*B == NX-1) || 
		(threadIdx.y+blockIdx.y*B == 0) || (threadIdx.y+blockIdx.y*B == NY-1) ||
		(threadIdx.z+blockIdx.z*B == 0) || (threadIdx.z+blockIdx.z*B == NZ-1)){
		vnew = 0;
	}

	//if(threadIdx.x == 2 && blockIdx.x == 1) printf("vnew = %e\n",vnew);

	// write results
	Vout[gidx] = vnew;
}

__global__ void __launch_bounds__(512, 4) gpu_hartree_iteration_crap(
	float* 		Q,			// density grid
	float*		V, 			// current V guess
	float*		Vout 		// updated V
	){

	// global memory address
	uint gidx = threadIdx.x + blockIdx.x*B + (threadIdx.y+blockIdx.y*B)*NX + (threadIdx.z+blockIdx.z*B)*NX*NY;
	int3 r;
	r.x = threadIdx.x + blockIdx.x*B;
	r.y = threadIdx.y+blockIdx.y*B;
	r.z = threadIdx.z+blockIdx.z*B;

	int flag = 1;
	float vnew = 0;

	if(r.x > 1) vnew += V[gidx-1];
	if(r.y > 1) vnew += V[gidx-NX];
	if(r.z > 1) vnew += V[gidx-NX*NY];

	if(r.x < NX-2) vnew += V[gidx+1];
	if(r.y < NY-2) vnew += V[gidx+NX];
	if(r.z < NZ-2) vnew += V[gidx+NX*NY];

	vnew = vnew*STEP*STEP/6.0f + Q[gidx]*(STEP)/6.0f;

	if((r.x == 0) || (r.x == NX-1) || 
		(r.y == 0) || (r.y == NY-1) ||
		(r.z == 0) || (r.z == NZ-1)){
		vnew = 0;
	}

	Vout[gidx] = vnew;
}


__global__ void __launch_bounds__(512, 4) gpu_hartree_add_nuclei(
	int* 		types,
	float3*		coords, 	// atom coordinates in BOHR
	float*		V 			// output hartree qube
	){

	__shared__ int styp[100];
	__shared__ float3 scoords[100];

	uint sidx = threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;

	if(sidx < NATM) {
		styp[sidx] = types[sidx];
		scoords[sidx] = coords[sidx];
	}
	__syncthreads();

	float hartree = 0;
	float c,r;

	// compute voxel position in the output V grid
	float3 V_pos;
	V_pos.x = X0 + (blockIdx.x * B + threadIdx.x) * STEP + 0.5f*STEP;
	V_pos.y = Y0 + (blockIdx.y * B + threadIdx.y) * STEP + 0.5f*STEP;
	V_pos.z = Z0 + (blockIdx.z * B + threadIdx.z) * STEP + 0.5f*STEP;


	// add the nuclear potential
	for(ushort i=0; i<NATM; i++) {

		c = V_pos.x - scoords[i].x; r = c*c;
		c = V_pos.y - scoords[i].y; r+= c*c;
		c = V_pos.z - scoords[i].z; r+= c*c;
		r = sqrtf(r);
		hartree += styp[i] / r;
	}
	
	// write results
	sidx = (threadIdx.x + blockIdx.x*B);
	sidx+= (threadIdx.y + blockIdx.y*B) * NX;
	sidx+= (threadIdx.z + blockIdx.z*B) * NX*NY;
	V[sidx] += hartree;
}


