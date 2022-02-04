#include "kernel_common.h"


#define NORBS PYCUDA_NORBS
#define NATOMS PYCUDA_NATOMS



__device__ float gpu_SolidHarmonicR(short L, short m, float3 r) {

	// no branching cos all threads call the same L,m with different r
	if(L == 0) return 0.28209479177387814f;
	else if(L == 1) {
		if(m == -1) return 0.4886025119029199f  * r.y;
		if(m ==  0) return 0.4886025119029199f  * r.z;
		if(m ==  1) return 0.4886025119029199f  * r.x;
	}
	else if(L == 2) {
		if(m == -2) return 1.0925484305920792f  * r.x * r.y;
		if(m == -1) return 1.0925484305920792f  * r.z * r.y;
		if(m ==  0) return 0.31539156525252005f * (2*r.z*r.z - r.x*r.x - r.y*r.y);
		if(m ==  1) return 1.0925484305920792f  * r.x * r.z;
		if(m ==  2) return 0.5462742152960396f  * (r.x*r.x - r.y*r.y);
	}
	return 0;
}


__global__ __launch_bounds__(512, 2) void gpu_densityqube_shmem_subgrid(
	float 		dx,		 	// grid step size - in BOHR
	float3 		grid0, 		// grid origin in BOHR
	float3*		coords, 	// atom coordinates in BOHR
	float*		alphas,
	float*		coeffs,
	short4*		almos,		// indexes of orbital properties
	float*		dm,			// density matrix
	float*		qube 		// output density cube
	){

	__shared__ float3 scoords[NATOMS];
	__shared__ float shDM[NORBS];
	volatile __shared__ short4 shALMOs[NORBS];

	volatile __shared__ float shALPHAp[MAXAOC];
	volatile __shared__ float shALPHAq[MAXAOC];
	volatile __shared__ float shCOEFFp[MAXAOC];
	volatile __shared__ float shCOEFFq[MAXAOC];


	volatile uint sidx = threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;

	if(sidx < NATOMS) {
		scoords[sidx] = coords[sidx];
		scoords[sidx].x -= grid0.x;
		scoords[sidx].y -= grid0.y;
		scoords[sidx].z -= grid0.z;
	}
	if(sidx < NORBS) {
		short4 t = almos[sidx];
		shALMOs[sidx].x = t.x;
		shALMOs[sidx].y = t.y;
		shALMOs[sidx].z = t.z;
		shALMOs[sidx].w = t.w;
	}
	__syncthreads();

	float charge = 0;

	for(ushort p=0; p<NORBS; ++p) { // loop over DM rows

		// load DM row in shmem
		if(sidx < NORBS) shDM[sidx] = dm[sidx + p*NORBS];

		// load info of p-th orbital
		if(sidx < MAXAOC) {
			shALPHAp[sidx] = alphas[shALMOs[p].w+sidx];
			shCOEFFp[sidx] = coeffs[shALMOs[p].w+sidx];
		}
		__syncthreads();


		for(ushort q=0; q<=p; ++q) { // loop over DM columns

			float3 voxpos;
			if(sidx < MAXAOC) {
				shALPHAq[sidx] = alphas[shALMOs[q].w+sidx];
				shCOEFFq[sidx] = coeffs[shALMOs[q].w+sidx];
			}
			__syncthreads();
			for(ushort ix=0; ix<SUBGRID; ix++) {
				voxpos.x = (blockIdx.x * B + threadIdx.x) * dx +ix*dx*SUBGRIDDX + SUBGRIDDX2;
				for(ushort iy=0; iy<SUBGRID; iy++) {
					voxpos.y = (blockIdx.y * B + threadIdx.y) * dx + iy*dx*SUBGRIDDX + SUBGRIDDX2;
					for(ushort iz=0; iz<SUBGRID; iz++) {
						voxpos.z = (blockIdx.z * B + threadIdx.z) * dx + iz*dx*SUBGRIDDX + SUBGRIDDX2;

						volatile float partial = shDM[q];
						volatile float3 r; // = scoords[shALMOs[p].x];
						r.x = voxpos.x - scoords[shALMOs[p].x].x;
						r.y = voxpos.y - scoords[shALMOs[p].x].y;
						r.z = voxpos.z - scoords[shALMOs[p].x].z;
						
						partial *= gpu_SolidHarmonicR(shALMOs[p].y, shALMOs[p].z, r);

						// multiply by the contracted gaussians
						r.x = r.x*r.x + r.y*r.y + r.z*r.z;
						r.y = 0;
						for(ushort ai=0; ai<MAXAOC; ai++) {
							//r.z = alphas[shALMOs[p].w+ai];
							r.y += shCOEFFp[ai] * exp(-shALPHAp[ai] * r.x);
						}
						partial *= r.y;

						//r = scoords[shALMOs[q].x];
						r.x = voxpos.x - scoords[shALMOs[q].x].x;
						r.y = voxpos.y - scoords[shALMOs[q].x].y;
						r.z = voxpos.z - scoords[shALMOs[q].x].z;
						partial *= gpu_SolidHarmonicR(shALMOs[q].y, shALMOs[q].z, r);

						r.x = r.x*r.x + r.y*r.y + r.z*r.z;
						r.y = 0;
						for(ushort ai=0; ai<MAXAOC; ai++) {
							//r.z = alphas[shALMOs[q].w+ai];
							r.y += shCOEFFq[ai] * exp(-shALPHAq[ai] * r.x);
						}
						partial *= r.y;

						if(p!=q) partial*=2;

						charge += partial * SUBGRIDiV;
					}
				}
			}// end of subgrid loop
			__syncthreads();
		}
		__syncthreads();

	}

	// this accounts for closed shell (2 electrons per orbital)
	// and multiplied by the voxel volume (integral of the wfn)
	charge = 2*charge*dx*dx*dx;

	// compute the write index

	sidx = (threadIdx.x + blockIdx.x*B);
	sidx+= (threadIdx.y + blockIdx.y*B) * gridDim.x * B;
	sidx+= (threadIdx.z + blockIdx.z*B) * gridDim.x * gridDim.y * B_2;
	qube[sidx] = charge;
}






__global__ void gpu_test(float3 a){

	
	printf("B[%i %i %i] T[%i %i %i] %f\n", blockIdx.x, blockIdx.y, blockIdx.z,
		threadIdx.x, threadIdx.y, threadIdx.z, a.x);
}