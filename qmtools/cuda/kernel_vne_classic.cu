#include "kernel_common.h"

#define NATOMS PYCUDA_NATOMS
#define NX PYCUDA_NX
#define NY PYCUDA_NY
#define NZ PYCUDA_NZ
#define DX PYCUDA_DX // this is the main grid step
#define NDIV PYCUDA_NDIV
#define dx PYCUDA_dx // this is the subdivision step


/// Performs a diffusion/generation/adsorption iteration of the nuclear pseudo-field.
__global__ void gpu_vne_classic(
	int 	*types,
	float3 	*coords,
	float 	*qout
	) {


	__shared__ float deltas[B_3];

	// shared atom types and pos
	__shared__ int styp[NATOMS];
	__shared__ float3 scrd[NATOMS];
	uint gidx = threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;
	if(gidx < NATOMS) {
		styp[gidx] = types[gidx];
		scrd[gidx] = coords[gidx];
		scrd[gidx].x -= PYCUDA_X0;
		scrd[gidx].y -= PYCUDA_Y0;
		scrd[gidx].z -= PYCUDA_Z0;
	}
	__syncthreads();
	// the atoms are now loaded!

	int3 r;
	uint ridx;
	r.x = threadIdx.x + blockIdx.x*B;
	r.y = threadIdx.y + blockIdx.y*B;
	r.z = threadIdx.z + blockIdx.z*B;
	ridx = r.x + r.y*NX + r.z*NX*NY;


	// compute voxel position
	float3 x0, xp;
	x0.x = r.x * DX;
	x0.y = r.y * DX;
	x0.z = r.z * DX;
	float vvox = 0;
	int cnt = 0;

	// loop over all voxel subdivisions
	for(uint ii=0;ii<NDIV;ii++){
		for(uint jj=0;jj<NDIV;jj++){
			for(uint kk=0;kk<NDIV;kk++){

				// position of this voxel subdivision
				xp.x = x0.x + (ii + 0.5)*dx;
				xp.y = x0.y + (jj + 0.5)*dx;
				xp.z = x0.z + (kk + 0.5)*dx;

				for(uint i=0; i<NATOMS; i++){

					// compute the distance between atom i and the voxel subdivision
					float d = (xp.x-scrd[i].x)*(xp.x-scrd[i].x) + (xp.y-scrd[i].y)*(xp.y-scrd[i].y) + (xp.z-scrd[i].z)*(xp.z-scrd[i].z);
					
					//skipped += (d==0) * 1;
					//d = (d==0)*0 + (d!=0)*rsqrtf(d);
					
					if(d != 0) {
						d = rsqrtf(d); // inverse distance
						vvox += styp[i] * d;
						cnt++;
					}
					__syncthreads();
				}
			}
		}
	}
	vvox /= cnt;
	vvox /= DX*DX*DX; // normalize by the voxel volume?

	qout[ridx] = vvox;
}
