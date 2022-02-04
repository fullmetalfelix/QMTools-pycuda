#include "kernel_common.h"

#define NATM PYCUDA_NATOMS

#define STEP PYCUDA_GRID_STEP
#define X0 PYCUDA_GRID_X0
#define Y0 PYCUDA_GRID_Y0
#define Z0 PYCUDA_GRID_Z0
#define NX PYCUDA_GRID_NX
#define NY PYCUDA_GRID_NY
#define NZ PYCUDA_GRID_NZ


#define VSTEP PYCUDA_VGRID_STEP
#define VX0 PYCUDA_VGRID_X0
#define VY0 PYCUDA_VGRID_Y0
#define VZ0 PYCUDA_VGRID_Z0




// direct integration kernel - debug function, ignores atomic nuclei
__global__ void __launch_bounds__(512, 4) gpu_hartree_noAtoms(
	float* 		q,			// density grid
	float*		V, 			// output hartree qube
	int* 		types,
	float3*		coords 		// atom coordinates in BOHR
){

	volatile __shared__ int styp[100];
	volatile __shared__ float3 scoords[100];
	volatile __shared__ float sQ[B_3];


	volatile uint sidx = threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;
	volatile uint ridx;

	if(sidx < NATM) {
		styp[sidx] = types[sidx];
		volatile float3 tmp = coords[sidx];
		scoords[sidx].x = tmp.x;
		scoords[sidx].y = tmp.y;
		scoords[sidx].z = tmp.z;
	}
	__syncthreads();

	volatile float hartree = 0;
	volatile float c,r;

	// compute voxel position in the output V grid
	volatile float3 V_pos;
	V_pos.x = VX0 + (blockIdx.x * B + threadIdx.x) * VSTEP + 0.5f*VSTEP;
	V_pos.y = VY0 + (blockIdx.y * B + threadIdx.y) * VSTEP + 0.5f*VSTEP;
	V_pos.z = VZ0 + (blockIdx.z * B + threadIdx.z) * VSTEP + 0.5f*VSTEP;


	// loop over the blocks of density grid
	for(ushort x=0; x<NX; ++x) {
		for(ushort y=0; y<NY; ++y) {
			for(ushort z=0; z<NZ; ++z) {

				// load a patch of Q grid
				
				ridx = x*B + threadIdx.x;
				ridx+=(y*B + threadIdx.y) * NX * B;
				ridx+=(z*B + threadIdx.z) * NX*NY*B_2;
				sQ[sidx] = q[ridx];
				__syncthreads();
				// now we have the patch... loop!

				for(uint sx=0; sx<B; ++sx) {
					for(uint sy=0; sy<B; ++sy) {
						for(uint sz=0; sz<B; ++sz) {

							//float c, r=0; // distance between the V evaluation point and the q voxel center

							c = V_pos.x - (X0 + (x*B + sx) * STEP + 0.5f*STEP);
							r = c*c;
							c = V_pos.y - (Y0 + (y*B + sy) * STEP + 0.5f*STEP);
							r+= c*c;
							c = V_pos.z - (Z0 + (z*B + sz) * STEP + 0.5f*STEP);
							r+= c*c;
							r = sqrtf(r);

							if(r < 0.5f*STEP) 
								hartree += (sQ[sx + sy*B + sz*B_2]/STEP) * (3.0f - 4.0f*r*r/(STEP*STEP)); // assume uniform charge sphere if V point is in the same voxel as Q
							else 
								hartree += sQ[sx + sy*B + sz*B_2] / r;
						}
					}
				}
				__syncthreads();
			}
		}
	}

	hartree = -hartree; // because electrons are negative!

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
	sidx+= (threadIdx.y + blockIdx.y*B) * gridDim.x * B;
	sidx+= (threadIdx.z + blockIdx.z*B) * gridDim.x * gridDim.y * B_2;
	V[sidx] = hartree;
}





/*
// direct integration kernel
__global__ void gpu_hartree(
	//float 		q_dx,	 	// density grid parameters
	float3 		q_x0, 		// 
	dim3 		q_n, 		// 
	float* 		q,			// density grid

	float 		V_dx, 		// hartree grid parameters
	float3 		V_x0,
	float*		V, 			// output hartree qube
	
	int 		natoms,		// number of atoms in the molecule
	int* 		types,
	float3*		coords 		// atom coordinates in BOHR
){

	__shared__ int styp[100];
	__shared__ float3 scoords[100];
	__shared__ float sQ[B_3];


	uint sidx = threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;

	if(sidx < natoms) {
		styp[sidx] = types[sidx];
		scoords[sidx] = coords[sidx];
	}
	__syncthreads();

	float hartree = 0;

	// compute voxel position in the output V grid
	float3 V_pos;
	V_pos.x = V_x0.x + (blockIdx.x * B + threadIdx.x) * V_dx + 0.5f*V_dx;
	V_pos.y = V_x0.y + (blockIdx.y * B + threadIdx.y) * V_dx + 0.5f*V_dx;
	V_pos.z = V_x0.z + (blockIdx.z * B + threadIdx.z) * V_dx + 0.5f*V_dx;


	// loop over the blocks of density grid
	for(ushort x=0; x<q_n.x; ++x) {
		for(ushort y=0; y<q_n.y; ++y) {
			for(ushort z=0; z<q_n.z; ++z) {

				// load a patch of Q grid
				uint ridx;
				ridx = x*B + threadIdx.x;
				ridx+=(y*B + threadIdx.y) * q_n.x * B;
				ridx+=(z*B + threadIdx.z) * q_n.x*q_n.y*B_2;
				sQ[sidx] = q[ridx];
				__syncthreads();
				// now we have the patch... loop!


				for(ushort sx=0; sx<B; ++sx) {
					for(ushort sy=0; sy<B; ++sy) {
						for(ushort sz=0; sz<B; ++sz) {

							float c, r=0; // distance between the V evaluation point and the q voxel center

							c = V_pos.x - (q_x0.x + (x*B + sx) * q_dx + 0.5f*q_dx);
							r = c*c;
							c = V_pos.y - (q_x0.y + (y*B + sy) * q_dx + 0.5f*q_dx);
							r+= c*c;
							c = V_pos.z - (q_x0.z + (z*B + sz) * q_dx + 0.5f*q_dx);
							r+= c*c;
							r = sqrtf(r);

							/*r.x = V_pos.x - (q_x0.x + (x*B + sx) * q_dx + 0.5f*q_dx);
							r.y = V_pos.y - (q_x0.y + (y*B + sy) * q_dx + 0.5f*q_dx);
							r.z = V_pos.z - (q_x0.z + (z*B + sz) * q_dx + 0.5f*q_dx);
							r.w = r.x*r.x + r.y*r.y + r.z*r.z;
							r.w = sqrtf(r.w);* /

							if(r < 0.5f*q_dx) 
								hartree += (sQ[sx + sy*B + sz*B_2]/q_dx) * (3.0f - 4.0f*r*r/(q_dx*q_dx));
							else 
								hartree += sQ[sx + sy*B + sz*B_2] / r;
						}
					}
				}
				__syncthreads();
			}
		}
	}

	hartree = -hartree; // because electrons are negative!

	// add the nuclear potential
	for(ushort i=0; i<natoms; i++) {

		float c, r=0; // distance between the V evaluation point and the q voxel center

		c = V_pos.x - scoords[i].x; r = c*c;
		c = V_pos.y - scoords[i].y; r+= c*c;
		c = V_pos.z - scoords[i].z; r+= c*c;
		r = sqrtf(r);
		hartree += styp[i] / r;
	}

	// write results
	sidx = (threadIdx.x + blockIdx.x*B);
	sidx+= (threadIdx.y + blockIdx.y*B) * gridDim.x * B;
	sidx+= (threadIdx.z + blockIdx.z*B) * gridDim.x * gridDim.y * B_2;
	V[sidx] = hartree;
}
*/

/*
void qm_hartree(Molecule *m, Grid *q, Grid *v) {

	cudaError_t cudaError;
	printf("computing hartree qube...\n");

	dim3 block(B,B,B);
	gpu_hartree<<<v->GPUblocks, block>>>(
		q->step,
		q->origin,
		q->GPUblocks,
		q->d_qube,

		v->step,
		v->origin,
		v->d_qube,

		m->natoms,
		m->d_types,
		m->d_coords
	);

	cudaDeviceSynchronize();
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess)
		printf("gpu_v_qube error: %s\n", cudaGetErrorString(cudaError));
	assert(cudaError == cudaSuccess);
	// TODO: THERE IS AN INVALID MEMORY ADDRESS IN THE KERNEL!?

	cudaError = cudaMemcpy(v->qube, v->d_qube, sizeof(float)*v->npts, cudaMemcpyDeviceToHost);assert(cudaError == cudaSuccess);
}
*/