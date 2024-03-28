#include "kernel_common.h"

#define NATM PYCUDA_NATOMS

#define STEP PYCUDA_GRID_STEP
#define X0 PYCUDA_GRID_X0
#define Y0 PYCUDA_GRID_Y0
#define Z0 PYCUDA_GRID_Z0
#define NX PYCUDA_GRID_NX
#define NY PYCUDA_GRID_NY
#define NZ PYCUDA_GRID_NZ

#define NORB 0

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



#define PI_3over2 5.568327996831707f
#define PI_1over2 1.7724538509055159f
#define PI 3.141592653589793f


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





// FULL ANALYTICAL METHOD



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








// only uses a slice of DM/ALMOs with s and p orbitals
__global__ void gpu_hartree_GTO(
	int* 		types,
	float3*		coords, 	// atom coordinates in BOHR
	float*		alphas,
	float*		coeffs,
	short4*		almos,		// indexes of orbital properties
	float*		dm,			// density matrix
	float*		Vout 		// output hartree cube
	){

	__shared__ int stypes[NATM];
	__shared__ float3 scoords[NATM];
	__shared__ float shDM[NORB];
	__shared__ short4 shALMOs[NORB];

	__shared__ short4 shALMOa, shALMOb;
	__shared__ float shALPHAa[MAXAOC];
	__shared__ float shALPHAb[MAXAOC];
	__shared__ float shCOEFFa[MAXAOC];
	__shared__ float shCOEFFb[MAXAOC];
	

	// shared a/b compute parameters
	volatile __shared__ float3 AB;
	volatile __shared__ float AB2, expAB, eta, gamma;


	uint sidx = threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;

	uint bidx = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
	bidx = (threadIdx.x + blockIdx.x*B);
	bidx+= (threadIdx.y + blockIdx.y*B) * gridDim.x * B;
	bidx+= (threadIdx.z + blockIdx.z*B) * gridDim.x * gridDim.y * B_2;


	// load atom positions and orbital ALMOs
	if(sidx < NATM) {
		scoords[sidx] = coords[sidx];
		scoords[sidx].x -= X0;
		scoords[sidx].y -= Y0;
		scoords[sidx].z -= Z0;
		stypes[sidx] = types[sidx];
	}
	if(sidx < NORB) {
		shALMOs[sidx] = almos[sidx];
		//shALMOs[sidx].x = t.x;
		//shALMOs[sidx].y = t.y;
		//shALMOs[sidx].z = t.z;
		//shALMOs[sidx].w = t.w;
	}
	__syncthreads();

	float V = 0;

	float3 rC; // this is the position where we compute potential
	rC.x = (blockIdx.x * B + threadIdx.x)*STEP;
	rC.y = (blockIdx.y * B + threadIdx.y)*STEP;
	rC.z = (blockIdx.z * B + threadIdx.z)*STEP;


	for(ushort p=0; p<NORB; ++p) { // loop over DM rows

		// load DM row in shmem
		if(sidx < NORB) shDM[sidx] = dm[sidx + p*NORB];
		__syncthreads();

		for(ushort q=0; q<=p; ++q) { // loop over DM columns

			//if(p != q) continue;

			// order the orbitals by L and m on thread 0
			if(sidx == 0){

				if(shALMOs[p].y < shALMOs[q].y) { // Lp < Lq => p->A q->B

					shALMOa = shALMOs[p];
					shALMOb = shALMOs[q];

				} else if(shALMOs[p].y == shALMOs[q].y){ // same L => check m
					// highest M goes last
					if(shALMOs[p].z <= shALMOs[q].z){
						shALMOa = shALMOs[p];
						shALMOb = shALMOs[q];
					} else {
						shALMOa = shALMOs[q];
						shALMOb = shALMOs[p];
					}
				} else {// Lp > Lq => p->B q->A

					shALMOa = shALMOs[q];
					shALMOb = shALMOs[p];
				}
			}
			__syncthreads();


			// load the params and coeffs for a/b
			if(sidx < MAXAOC) {
				shALPHAa[sidx] = alphas[shALMOa.w+sidx];
				shCOEFFa[sidx] = coeffs[shALMOa.w+sidx];

				shALPHAb[sidx] = alphas[shALMOb.w+sidx];
				shCOEFFb[sidx] = coeffs[shALMOb.w+sidx];
			}
			__syncthreads();



			if(sidx == 0){
				// this is the (A-B) vector
				AB.x = scoords[shALMOa.x].x - scoords[shALMOb.x].x;
				AB.y = scoords[shALMOa.x].y - scoords[shALMOb.x].y;
				AB.z = scoords[shALMOa.x].z - scoords[shALMOb.x].z;
				AB2 = (AB.x*AB.x + AB.y*AB.y + AB.z*AB.z);
			}__syncthreads();

			//if(AB2 > 36) continue;

			float product = 0;

			for(ushort ai=0; ai<MAXAOC; ai++){
				if(shCOEFFa[ai] == 0) break;
				for(ushort bi=0; bi<MAXAOC; bi++){
					if(shCOEFFb[bi]==0) break;

					if(sidx == 0) {
						gamma = shALPHAa[ai] + shALPHAb[bi];
						eta = shALPHAa[ai]*shALPHAb[bi] / gamma;
						expAB = expf(-eta * AB2);
					}
					__syncthreads();

					// this is P-r_c
					float3 rPC;
					rPC.x = (shALPHAa[ai]*scoords[shALMOa.x].x + shALPHAb[bi]*scoords[shALMOb.x].x)/gamma - rC.x;
					rPC.y = (shALPHAa[ai]*scoords[shALMOa.x].y + shALPHAb[bi]*scoords[shALMOb.x].y)/gamma - rC.y;
					rPC.z = (shALPHAa[ai]*scoords[shALMOa.x].z + shALPHAb[bi]*scoords[shALMOb.x].z)/gamma - rC.z;

					float PC = rPC.x*rPC.x + rPC.y*rPC.y + rPC.z*rPC.z;
					float ERFgPC = erff(sqrtf(gamma*PC));
					float PC2gamma = PC * gamma;

					float factor = 0;

					if(shALMOa.y == 0 && shALMOb.y == 0){ // S-S case
						
						factor = - 0.28209479177387814*0.28209479177387814*(ERFgPC*PI_3over2)/(sqrtf(PC2gamma)*gamma);

					} else if (shALMOa.y == 0 && shALMOb.y == 1) { // S-P

						
						float ABx = AB.z;
						float PCx = rPC.z;
						if(shALMOb.z == -1) {ABx = AB.y; PCx = rPC.y;}
						else if(shALMOb.z == 1) {ABx = AB.x; PCx = rPC.x;}

						factor = ERFgPC*PI_1over2*(PCx*shALPHAb[bi] - 2*PC2gamma*ABx*gamma*eta);
						factor -= 2*sqrtf(PC2gamma)*PCx*shALPHAb[bi] * expf(-PC2gamma);
						factor *= PI / (2*shALPHAb[bi]);

						ABx = PC2gamma*gamma*sqrtf(PC2gamma);
						factor *= 0.4886025119029199 * 0.28209479177387814 / ABx;

					} else if (shALMOa.y == 1 && shALMOb.y == 1) { // both P

						
						float ABx = AB.z;
						float PCx = rPC.z;
						if(shALMOa.z == -1) {ABx = AB.y; PCx = rPC.y;}
						else if(shALMOa.z == 1) {ABx = AB.x; PCx = rPC.x;}

						if(shALMOa.z == shALMOb.z){ // P?-P? (same m)


							// this first line causes grief!
							factor = ERFgPC*PI_1over2*(-3*PCx*PCx + PC*(1 + 2*ABx*PCx*(shALPHAa[ai]-shALPHAb[bi]) + PC2gamma*(4*ABx*ABx*eta-2) + 1));
							factor+= 2*expf(-PC2gamma)*sqrtf(PC2gamma)*((3+2*PC2gamma)*PCx*PCx + PC*(-1+2*ABx*PCx*(-shALPHAa[ai]+shALPHAb[bi])));
							
							
							ABx = 4*PC2gamma*PC2gamma*sqrtf(PC2gamma)*gamma;
							factor *= PI*0.4886025119029199*0.4886025119029199 / ABx;
							
							//if(bidx == 0){
								//printf("%f %f %f %f -- %f %f -- %e --  AB2 %f \n",eta,gamma,PC,expAB,   ABx,PCx,   factor,    AB2);
							//}
							
						} else { // different m
							
							float ABy = AB.z;
							float PCy = rPC.z;
							if(shALMOb.z == -1) {ABy = AB.y; PCy = rPC.y;}
							else if(shALMOb.z == 1) {ABy = AB.x; PCy = rPC.x;}
							
							factor = 2*sqrtf(PC2gamma)*expf(-PC2gamma)*(PCx*(3*PCy + 2*PC2gamma*PCy - 2*ABy*PC*shALPHAa[ai]) + 2*ABx*PC*PCy*shALPHAb[bi]);
							factor-= ERFgPC*PI_1over2*(PCx*(3*PCy - 2*ABy*PC*shALPHAa[ai]) + 2*ABx*PC*(PCy*shALPHAb[bi] - 2*ABy*PC2gamma*eta));

							ABx = 4*PC2gamma*PC2gamma*sqrtf(PC2gamma)*gamma;
							factor *= PI*0.4886025119029199*0.4886025119029199 / ABx;

						}
						
	
					} else if (shALMOa.y == 0 && shALMOb.y == 2) { // S-D

						
						if(shALMOb.z < 0 || shALMOb.z == 1){ // Dxy Dxz Dyz

							// -2 => xy, -1 => yz, 1 => xz

							float ABx = AB.x,  ABy = AB.z;
							float PCx = rPC.x, PCy = rPC.z;
							if(shALMOb.z == -2){ABy = AB.y; PCy = rPC.y;}
							else if(shALMOb.z == -1) {ABx = AB.y; PCx = rPC.y;}

							factor = (2*sqrtf(PC2gamma)*shALPHAb[bi]*(-2*ABx*PC2gamma*PCy*eta + PCx*((3 + 2*PC2gamma)*PCy*shALPHAb[bi] - 2*ABy*PC2gamma*eta)))*expf(-PC2gamma);
							factor+= ERFgPC*PI_1over2*(PCx*shALPHAb[bi]*(3*PCy*shALPHAb[bi] - 2*ABy*PC2gamma*eta) + 2*ABx*PC2gamma*eta*(-(PCy*shALPHAb[bi]) + 2*ABy*PC2gamma*eta));
							factor *= 1.0925484305920792;
						}
						else if(shALMOb.z == 0){ // D(2zz-xx-yy)

							float PCxxyym2zz = (rPC.x*rPC.x + rPC.y*rPC.y - 2*rPC.z*rPC.z);
							float ABxxyym2zz = (AB.x*AB.x + AB.y*AB.y - 2*AB.z*AB.z);

							factor = -2*expf(-PC2gamma)*sqrtf(PC2gamma)*shALPHAb[bi]*((3 + 2*PC2gamma)*PCxxyym2zz*shALPHAb[bi] + 2*(3 + 2*PC2gamma)*rPC.y*rPC.y*shALPHAb[bi] - 4*AB.y*PC2gamma*rPC.y*eta - 4*PC2gamma*(AB.x*rPC.x - 2*AB.z*rPC.z)*eta);
							factor+= ERFgPC*PI_1over2*(3*PCxxyym2zz*shALPHAb[bi]*shALPHAb[bi] + 6*rPC.y*rPC.y*shALPHAb[bi]*shALPHAb[bi] - 4*AB.y*PC2gamma*rPC.y*shALPHAb[bi]*eta + 4*PC2gamma*eta*(-(AB.x*rPC.x*shALPHAb[bi]) + 2*AB.z*rPC.z*shALPHAb[bi] + (ABxxyym2zz + 2*AB.y*AB.y)*PC2gamma*eta));

							factor *= 0.31539156525252005;
						}
						else if(shALMOb.z == 2) { // D(xx-yy)

							float PCxxmyy = rPC.x*rPC.x - rPC.y*rPC.y;
							float ABxxmyy = AB.x*AB.x - AB.y*AB.y;

							factor = (2*sqrtf(PC2gamma)*shALPHAb[bi]*((3 + 2*PC2gamma)*PCxxmyy*shALPHAb[bi] + 4*PC2gamma*(-(AB.x*rPC.x) + AB.y*rPC.y)*eta))*expf(-PC2gamma);
							factor-= ERFgPC*PI_1over2*(3*PCxxmyy*shALPHAb[bi]*shALPHAb[bi] + 4*PC2gamma*eta*(-(AB.x*rPC.x*shALPHAb[bi]) + AB.y*rPC.y*shALPHAb[bi] + ABxxmyy*PC2gamma*eta));
							factor *= 0.5462742152960396;
						}

						factor /= 4 * shALPHAb[bi]* shALPHAb[bi]*PC2gamma*PC2gamma*sqrtf(PC2gamma)*gamma;
						factor *= 0.28209479177387814*PI;
						
					} else if (shALMOa.y == 1 && shALMOb.y == 2) { // P-D

						// THE POT OF DOOM

						float PC2gamma2 = PC2gamma*PC2gamma;
						float PC2gamma3 = PC2gamma2*PC2gamma;
						float PCxxyym2zz = (rPC.x*rPC.x + rPC.y*rPC.y - 2*rPC.z*rPC.z);
						float ABxxyym2zz = (AB.x*AB.x + AB.y*AB.y - 2*AB.z*AB.z);
						float PCxxmyy = rPC.x*rPC.x - rPC.y*rPC.y;
						float ABxxmyy = AB.x*AB.x - AB.y*AB.y;


						if(shALMOa.z == -1 && shALMOb.z == -2) {

							factor = ERFgPC*PI_1over2*(15*rPC.x*rPC.y*rPC.y*shALPHAb[bi]*gamma + 4*AB.x*PC2gamma3*eta*(-1 + 2*AB.y*AB.y*eta) + PC2gamma2*(2*rPC.x*shALPHAb[bi] + 2*AB.x*eta + 4*AB.x*AB.y*rPC.y*shALPHAa[ai]*eta - 4*AB.y*(AB.y*rPC.x + AB.x*rPC.y)*shALPHAb[bi]*eta) + PC2gamma*(-3*rPC.x*shALPHAb[bi] + 6*AB.y*rPC.x*rPC.y*shALPHAb[bi]*shALPHAb[bi] - 6*rPC.y*(AB.y*rPC.x + AB.x*rPC.y)*gamma*eta));
							factor+= (sqrtf(PC2gamma)*gamma*(PC*(-4*AB.x*PC2gamma*(1 + 2*AB.y*rPC.y*(shALPHAa[ai] - shALPHAb[bi]))*eta + 2*rPC.x*shALPHAb[bi]*(3 - 2*AB.y*(3 + 2*PC2gamma)*rPC.y*shALPHAb[bi] + 4*AB.y*AB.y*PC2gamma*eta)) - 2*rPC.y*(-2*AB.x*PC2gamma*(3 + 2*PC2gamma)*rPC.y*eta + rPC.x*((15 + 10*PC2gamma + 4*PC2gamma2)*rPC.y*shALPHAb[bi] - 2*AB.y*PC2gamma*(3 + 2*PC2gamma)*eta))))*expf(-PC2gamma);

							factor *= (PI*eta)/(8*PC2gamma3*sqrtf(PC2gamma)*shALPHAa[ai]*shALPHAb[bi]*shALPHAb[bi]*gamma);
							factor *= 0.488603*1.09255;
						}
						else if(shALMOa.z == -1 && shALMOb.z == -1) {

							factor = ERFgPC*PI_1over2*(15*rPC.y*rPC.y*rPC.z*shALPHAb[bi]*gamma + 4*AB.z*PC2gamma3*eta*(-1 + 2*AB.y*AB.y*eta) + PC2gamma2*(2*rPC.z*shALPHAb[bi] + 2*AB.z*eta + 4*AB.y*AB.z*rPC.y*shALPHAa[ai]*eta - 4*AB.y*(AB.z*rPC.y + AB.y*rPC.z)*shALPHAb[bi]*eta) + PC2gamma*(-3*rPC.z*shALPHAb[bi] + 6*AB.y*rPC.y*rPC.z*shALPHAb[bi]*shALPHAb[bi] - 6*rPC.y*(AB.z*rPC.y + AB.y*rPC.z)*gamma*eta));
							factor+= (sqrtf(PC2gamma)*gamma*(PC*(-4*AB.z*PC2gamma*(1 + 2*AB.y*rPC.y*(shALPHAa[ai] - shALPHAb[bi]))*eta + 2*rPC.z*shALPHAb[bi]*(3 - 2*AB.y*(3 + 2*PC2gamma)*rPC.y*shALPHAb[bi] + 4*AB.y*AB.y*PC2gamma*eta)) - 2*rPC.y*(-2*AB.y*PC2gamma*(3 + 2*PC2gamma)*rPC.z*eta + rPC.y*((15 + 10*PC2gamma + 4*PC2gamma2)*rPC.z*shALPHAb[bi] - 2*AB.z*PC2gamma*(3 + 2*PC2gamma)*eta))))*expf(-PC2gamma);


							factor *= (PI*eta)/(8*PC2gamma3*sqrtf(PC2gamma)*shALPHAa[ai]*shALPHAb[bi]*shALPHAb[bi]*gamma);
							factor *= 0.488603*1.09255;
						}
						else if(shALMOa.z == -1 && shALMOb.z == 0) {

							factor = -(ERFgPC*PI_1over2*(15*PCxxyym2zz*rPC.y*shALPHAb[bi]*gamma + 8*AB.y*PC2gamma3*eta*(-1 + ABxxyym2zz*eta) + 4*PC2gamma2*(rPC.y*shALPHAb[bi] + AB.y*eta + ABxxyym2zz*rPC.y*shALPHAa[ai]*eta - 2*AB.y*(AB.x*rPC.x + AB.y*rPC.y - 2*AB.z*rPC.z)*shALPHAb[bi]*eta) + PC2gamma*(-6*rPC.y*shALPHAb[bi] + 6*AB.y*PCxxyym2zz*shALPHAb[bi]*shALPHAb[bi] - 12*rPC.y*(AB.x*rPC.x + AB.y*rPC.y - 2*AB.z*rPC.z)*gamma*eta)));
							factor+= (2*sqrtf(PC2gamma)*(PC*(-6*rPC.y*shALPHAb[bi] + 6*AB.y*PCxxyym2zz*shALPHAb[bi]*shALPHAb[bi] + 4*AB.y*PC2gamma*PCxxyym2zz*shALPHAb[bi]*shALPHAb[bi] + 4*AB.y*PC2gamma*eta + 4*ABxxyym2zz*PC2gamma*rPC.y*shALPHAa[ai]*eta - 8*AB.y*PC2gamma*(AB.x*rPC.x + AB.y*rPC.y - 2*AB.z*rPC.z)*shALPHAb[bi]*eta) + rPC.y*((15 + 10*PC2gamma + 4*PC2gamma2)*rPC.x*rPC.x*shALPHAb[bi] + (15 + 10*PC2gamma + 4*PC2gamma2)*rPC.y*rPC.y*shALPHAb[bi] - 4*AB.x*PC2gamma*(3 + 2*PC2gamma)*rPC.x*eta - 4*AB.y*PC2gamma*(3 + 2*PC2gamma)*rPC.y*eta - 2*rPC.z*((15 + 10*PC2gamma + 4*PC2gamma2)*rPC.z*shALPHAb[bi] - 4*AB.z*PC2gamma*(3 + 2*PC2gamma)*eta))))*expf(-PC2gamma);
							
							factor *= (PI*eta)/(8*PC2gamma3*sqrtf(PC2gamma)*shALPHAa[ai]*shALPHAb[bi]*shALPHAb[bi]*gamma);
							factor *= 0.488603*0.315392;
						}
						else if(shALMOa.z == -1 && shALMOb.z == 1) {

							factor = ERFgPC*PI_1over2*(2*AB.x*PC2gamma*eta*(rPC.y*(-3*rPC.z + 2*AB.z*PC*shALPHAa[ai]) - 2*AB.y*PC*(rPC.z*shALPHAb[bi] - 2*AB.z*PC2gamma*eta)) + rPC.x*(2*AB.y*PC*shALPHAb[bi]*(3*rPC.z*shALPHAb[bi] - 2*AB.z*PC2gamma*eta) + 3*rPC.y*(5*rPC.z*shALPHAb[bi] - 2*AB.z*PC2gamma*eta)));
							factor+= (sqrtf(PC2gamma)*(4*AB.x*PC2gamma*(rPC.y*(3*rPC.z + 2*PC2gamma*rPC.z - 2*AB.z*PC*shALPHAa[ai]) + 2*AB.y*PC*rPC.z*shALPHAb[bi])*eta - 2*rPC.x*(2*AB.y*PC*shALPHAb[bi]*((3 + 2*PC2gamma)*rPC.z*shALPHAb[bi] - 2*AB.z*PC2gamma*eta) + rPC.y*((15 + 10*PC2gamma + 4*PC2gamma2)*rPC.z*shALPHAb[bi] - 2*AB.z*PC2gamma*(3 + 2*PC2gamma)*eta))))*expf(-PC2gamma);
							
							factor *= (PI)/(8*PC2gamma3*sqrtf(PC2gamma)*shALPHAb[bi]*gamma);
							factor *= 0.488603*0.315392;
						}
						else if(shALMOa.z == -1 && shALMOb.z == 2) {

							factor = -(ERFgPC*PI_1over2*(15*rPC.y*(-rPC.x*rPC.x + rPC.y*rPC.y)*shALPHAb[bi]*gamma + 8*AB.y*PC2gamma3*eta*(-1 - AB.x*AB.x*eta + AB.y*AB.y*eta) + 4*PC2gamma2*(rPC.y*shALPHAb[bi] + AB.y*eta + (-AB.x*AB.x + AB.y*AB.y)*rPC.y*shALPHAa[ai]*eta + 2*AB.y*(AB.x*rPC.x - AB.y*rPC.y)*shALPHAb[bi]*eta) + 6*PC2gamma*(-(rPC.y*shALPHAb[bi]) + AB.y*(-rPC.x*rPC.x + rPC.y*rPC.y)*shALPHAb[bi]*shALPHAb[bi] + 2*rPC.y*(AB.x*rPC.x - AB.y*rPC.y)*gamma*eta)));
							factor+= (sqrtf(PC2gamma)*gamma*(2*rPC.y*((-15 - 10*PC2gamma - 4*PC2gamma2)*rPC.x*rPC.x*shALPHAb[bi] + 4*AB.x*PC2gamma*(3 + 2*PC2gamma)*rPC.x*eta + rPC.y*((15 + 10*PC2gamma + 4*PC2gamma2)*rPC.y*shALPHAb[bi] - 4*AB.y*PC2gamma*(3 + 2*PC2gamma)*eta)) + 4*PC*(AB.y*(3 + 2*PC2gamma)*rPC.y*rPC.y*shALPHAb[bi]*shALPHAb[bi] + AB.y*((-3 - 2*PC2gamma)*rPC.x*rPC.x*shALPHAb[bi]*shALPHAb[bi] + 2*PC2gamma*eta + 4*AB.x*PC2gamma*rPC.x*shALPHAb[bi]*eta) - rPC.y*(2*(AB.x*AB.x - AB.y*AB.y)*PC2gamma*shALPHAa[ai]*eta + shALPHAb[bi]*(3 + 4*AB.y*AB.y*PC2gamma*eta)))))*expf(-PC2gamma);
							
							factor *= (PI*eta)/(8*PC2gamma3*sqrtf(PC2gamma)*shALPHAa[ai]*shALPHAb[bi]*shALPHAb[bi]*gamma);
							factor *= 0.488603*0.546274;

						}



						else if(shALMOa.z == 0 && shALMOb.z == -2) {

							factor = ERFgPC*PI_1over2*(rPC.x*(3*rPC.y*shALPHAb[bi]*(5*rPC.z + 2*AB.z*PC*shALPHAb[bi]) - 2*AB.y*PC2gamma*(3*rPC.z + 2*AB.z*PC*shALPHAb[bi])*eta) + 2*AB.x*PC2gamma*eta*(-(rPC.y*(3*rPC.z + 2*AB.z*PC*shALPHAb[bi])) + 2*AB.y*PC*(rPC.z*shALPHAa[ai] + 2*AB.z*PC2gamma*eta)));
							factor+= (sqrtf(PC2gamma)*(4*AB.x*PC2gamma*(-2*AB.y*PC*rPC.z*shALPHAa[ai] + rPC.y*(3*rPC.z + 2*PC2gamma*rPC.z + 2*AB.z*PC*shALPHAb[bi]))*eta - 2*rPC.x*(rPC.y*shALPHAb[bi]*((15 + 10*PC2gamma + 4*PC2gamma2)*rPC.z + 2*AB.z*PC*(3 + 2*PC2gamma)*shALPHAb[bi]) - 2*AB.y*PC2gamma*((3 + 2*PC2gamma)*rPC.z + 2*AB.z*PC*shALPHAb[bi])*eta)))*expf(-PC2gamma);

							factor *= (PI)/(8*PC2gamma3*sqrtf(PC2gamma)*shALPHAb[bi]*gamma);
							factor*= 0.488603*1.09255;

						}
						else if(shALMOa.z == 0 && shALMOb.z == -1) {

							factor = ERFgPC*PI_1over2*(15*rPC.y*rPC.z*rPC.z*shALPHAb[bi]*gamma + 4*AB.y*PC2gamma3*eta*(-1 + 2*AB.z*AB.z*eta) + PC2gamma2*(2*rPC.y*shALPHAb[bi] + 2*AB.y*eta + 4*AB.y*AB.z*rPC.z*shALPHAa[ai]*eta - 4*AB.z*(AB.z*rPC.y + AB.y*rPC.z)*shALPHAb[bi]*eta) + PC2gamma*(-3*rPC.y*shALPHAb[bi] + 6*AB.z*rPC.y*rPC.z*shALPHAb[bi]*shALPHAb[bi] - 6*rPC.z*(AB.z*rPC.y + AB.y*rPC.z)*gamma*eta));
							factor+= (sqrtf(PC2gamma)*gamma*(PC*(-4*AB.y*PC2gamma*(1 + 2*AB.z*rPC.z*(shALPHAa[ai] - shALPHAb[bi]))*eta + 2*rPC.y*shALPHAb[bi]*(3 - 2*AB.z*(3 + 2*PC2gamma)*rPC.z*shALPHAb[bi] + 4*AB.z*AB.z*PC2gamma*eta)) - 2*rPC.z*(-2*AB.y*PC2gamma*(3 + 2*PC2gamma)*rPC.z*eta + rPC.y*((15 + 10*PC2gamma + 4*PC2gamma2)*rPC.z*shALPHAb[bi] - 2*AB.z*PC2gamma*(3 + 2*PC2gamma)*eta))))*expf(-PC2gamma);
							factor *= (PI*eta)/(8*PC2gamma3*sqrtf(PC2gamma)*shALPHAa[ai]*shALPHAb[bi]*shALPHAb[bi]*gamma);
							factor *= 0.488603*1.09255;

						}
						else if(shALMOa.z == 0 && shALMOb.z == 0) {

							factor = ERFgPC*PI_1over2*(-15*PCxxyym2zz*rPC.z*shALPHAb[bi]*gamma + 8*AB.z*PC2gamma3*eta*(-2 - ABxxyym2zz*eta) + PC2gamma2*(8*rPC.z*shALPHAb[bi] + 8*AB.z*eta - 4*ABxxyym2zz*rPC.z*shALPHAa[ai]*eta + 8*AB.z*(AB.x*rPC.x + AB.y*rPC.y - 2*AB.z*rPC.z)*shALPHAb[bi]*eta) + PC2gamma*(-12*rPC.z*shALPHAb[bi] - 6*AB.z*PCxxyym2zz*shALPHAb[bi]*shALPHAb[bi] + 12*rPC.z*(AB.x*rPC.x + AB.y*rPC.y - 2*AB.z*rPC.z)*gamma*eta));
							factor+= (2*sqrtf(PC2gamma)*gamma*(PC*(12*rPC.z*shALPHAb[bi] + 6*AB.z*PCxxyym2zz*shALPHAb[bi]*shALPHAb[bi] + 4*AB.z*PC2gamma*PCxxyym2zz*shALPHAb[bi]*shALPHAb[bi] - 8*AB.z*PC2gamma*eta + 4*ABxxyym2zz*PC2gamma*rPC.z*shALPHAa[ai]*eta - 8*AB.z*PC2gamma*(AB.x*rPC.x + AB.y*rPC.y - 2*AB.z*rPC.z)*shALPHAb[bi]*eta) + rPC.z*((15 + 10*PC2gamma + 4*PC2gamma2)*rPC.x*rPC.x*shALPHAb[bi] + (15 + 10*PC2gamma + 4*PC2gamma2)*rPC.y*rPC.y*shALPHAb[bi] - 4*AB.x*PC2gamma*(3 + 2*PC2gamma)*rPC.x*eta - 4*AB.y*PC2gamma*(3 + 2*PC2gamma)*rPC.y*eta - 2*rPC.z*((15 + 10*PC2gamma + 4*PC2gamma2)*rPC.z*shALPHAb[bi] - 4*AB.z*PC2gamma*(3 + 2*PC2gamma)*eta))))*expf(-PC2gamma);

							factor *= (PI*eta)/(8*PC2gamma3*sqrtf(PC2gamma)*shALPHAa[ai]*shALPHAb[bi]*shALPHAb[bi]*gamma);
							factor *= 0.488603*0.315392;

						}
						else if(shALMOa.z == 0 && shALMOb.z == 1) {

							factor = ERFgPC*PI_1over2*(15*rPC.x*rPC.z*rPC.z*shALPHAb[bi]*gamma + 4*AB.x*PC2gamma3*eta*(-1 + 2*AB.z*AB.z*eta) + PC2gamma2*(2*rPC.x*shALPHAb[bi] + 2*AB.x*eta + 4*AB.x*AB.z*rPC.z*shALPHAa[ai]*eta - 4*AB.z*(AB.z*rPC.x + AB.x*rPC.z)*shALPHAb[bi]*eta) + PC2gamma*(-3*rPC.x*shALPHAb[bi] + 6*AB.z*rPC.x*rPC.z*shALPHAb[bi]*shALPHAb[bi] - 6*rPC.z*(AB.z*rPC.x + AB.x*rPC.z)*gamma*eta));
							factor+= (sqrtf(PC2gamma)*gamma*(PC*(-4*AB.x*PC2gamma*(1 + 2*AB.z*rPC.z*(shALPHAa[ai] - shALPHAb[bi]))*eta + 2*rPC.x*shALPHAb[bi]*(3 - 2*AB.z*(3 + 2*PC2gamma)*rPC.z*shALPHAb[bi] + 4*AB.z*AB.z*PC2gamma*eta)) - 2*rPC.z*(-2*AB.x*PC2gamma*(3 + 2*PC2gamma)*rPC.z*eta + rPC.x*((15 + 10*PC2gamma + 4*PC2gamma2)*rPC.z*shALPHAb[bi] - 2*AB.z*PC2gamma*(3 + 2*PC2gamma)*eta))))*expf(-PC2gamma);

							factor *= (PI*eta)/(8*PC2gamma3*sqrtf(PC2gamma)*shALPHAa[ai]*shALPHAb[bi]*shALPHAb[bi]*gamma);
							factor *= 0.488603*0.315392;

						}
						else if(shALMOa.z == 0 && shALMOb.z == 2) {

							factor = ERFgPC*PI_1over2*(rPC.z*(15*PCxxmyy*shALPHAb[bi]*gamma + 4*ABxxmyy*PC2gamma2*shALPHAa[ai]*eta + 12*PC2gamma*(-(AB.x*rPC.x) + AB.y*rPC.y)*gamma*eta) + PC2gamma*(6*AB.z*PCxxmyy*shALPHAb[bi]*shALPHAb[bi] + PC2gamma*(-8*AB.x*AB.z*rPC.x + 8*AB.y*AB.z*rPC.y)*shALPHAb[bi]*eta + 8*ABxxmyy*AB.z*PC2gamma2*eta*eta));
							factor+= (2*sqrtf(PC2gamma)*gamma*(-(rPC.x*rPC.x*shALPHAb[bi]*((15 + 10*PC2gamma + 4*PC2gamma2)*rPC.z + 2*AB.z*PC*(3 + 2*PC2gamma)*shALPHAb[bi])) + rPC.y*rPC.y*shALPHAb[bi]*((15 + 10*PC2gamma + 4*PC2gamma2)*rPC.z + 2*AB.z*PC*(3 + 2*PC2gamma)*shALPHAb[bi]) - 4*ABxxmyy*PC*PC2gamma*rPC.z*shALPHAa[ai]*eta + 4*AB.x*PC2gamma*rPC.x*((3 + 2*PC2gamma)*rPC.z + 2*AB.z*PC*shALPHAb[bi])*eta - 4*AB.y*PC2gamma*rPC.y*((3 + 2*PC2gamma)*rPC.z + 2*AB.z*PC*shALPHAb[bi])*eta))*expf(-PC2gamma);

							factor *= (PI*eta)/(8*PC2gamma3*sqrtf(PC2gamma)*shALPHAa[ai]*shALPHAb[bi]*shALPHAb[bi]*gamma);
							factor*= 0.488603*0.546274;

						}




						else if(shALMOa.z == 1 && shALMOb.z == -2) {

							factor = ERFgPC*PI_1over2*(15*rPC.x*rPC.x*rPC.y*shALPHAb[bi]*gamma + 4*AB.y*PC2gamma3*eta*(-1 + 2*AB.x*AB.x*eta) + PC2gamma2*(2*rPC.y*shALPHAb[bi] + 2*AB.y*eta + 4*AB.x*AB.y*rPC.x*shALPHAa[ai]*eta - 4*AB.x*(AB.y*rPC.x + AB.x*rPC.y)*shALPHAb[bi]*eta) + PC2gamma*(-3*rPC.y*shALPHAb[bi] + 6*AB.x*rPC.x*rPC.y*shALPHAb[bi]*shALPHAb[bi] - 6*rPC.x*(AB.y*rPC.x + AB.x*rPC.y)*gamma*eta));
							factor+= (sqrtf(PC2gamma)*gamma*(PC*(-4*AB.y*PC2gamma*(1 + 2*AB.x*rPC.x*(shALPHAa[ai] - shALPHAb[bi]))*eta + 2*rPC.y*shALPHAb[bi]*(3 - 2*AB.x*(3 + 2*PC2gamma)*rPC.x*shALPHAb[bi] + 4*AB.x*AB.x*PC2gamma*eta)) - 2*rPC.x*(-2*AB.x*PC2gamma*(3 + 2*PC2gamma)*rPC.y*eta + rPC.x*((15 + 10*PC2gamma + 4*PC2gamma2)*rPC.y*shALPHAb[bi] - 2*AB.y*PC2gamma*(3 + 2*PC2gamma)*eta))))*expf(-PC2gamma);

							factor *= (PI*eta)/(8*PC2gamma3*sqrtf(PC2gamma)*shALPHAa[ai]*shALPHAb[bi]*shALPHAb[bi]*gamma);
							factor *= 0.488603*1.09255;

						}
						else if(shALMOa.z == 1 && shALMOb.z == -1) {

							factor = ERFgPC*PI_1over2*(rPC.x*(15*rPC.y*rPC.z*shALPHAb[bi] - 6*PC2gamma*(AB.z*rPC.y + AB.y*rPC.z)*eta + 4*AB.y*AB.z*PC*PC2gamma*shALPHAa[ai]*eta) + PC*(6*AB.x*rPC.y*rPC.z*shALPHAb[bi]*shALPHAb[bi] - 4*AB.x*PC2gamma*(AB.z*rPC.y + AB.y*rPC.z)*shALPHAb[bi]*eta + 8*AB.x*AB.y*AB.z*PC2gamma2*eta*eta));
							factor+= (sqrtf(PC2gamma)*(-4*AB.x*PC*shALPHAb[bi]*(3*rPC.y*rPC.z*shALPHAb[bi] + 2*PC2gamma*rPC.y*rPC.z*shALPHAb[bi] - 2*PC2gamma*(AB.z*rPC.y + AB.y*rPC.z)*eta) - 2*rPC.x*(-2*AB.y*PC2gamma*((3 + 2*PC2gamma)*rPC.z - 2*AB.z*PC*shALPHAa[ai])*eta + rPC.y*((15 + 10*PC2gamma + 4*PC2gamma2)*rPC.z*shALPHAb[bi] - 2*AB.z*PC2gamma*(3 + 2*PC2gamma)*eta))))*expf(-PC2gamma);

							factor *= (PI)/(8*PC2gamma3*sqrtf(PC2gamma)*shALPHAb[bi]*gamma);
							factor *= 0.488603*1.09255;

						}
						else if(shALMOa.z == 1 && shALMOb.z == 0) {

							factor = -(ERFgPC*PI_1over2*(15*rPC.x*PCxxyym2zz*shALPHAb[bi]*gamma + 8*AB.x*PC2gamma3*eta*(-1 + ABxxyym2zz*eta) + 4*PC2gamma2*(rPC.x*shALPHAb[bi] + AB.x*eta + ABxxyym2zz*rPC.x*shALPHAa[ai]*eta - 2*AB.x*(AB.x*rPC.x + AB.y*rPC.y - 2*AB.z*rPC.z)*shALPHAb[bi]*eta) + PC2gamma*(-6*rPC.x*shALPHAb[bi] + 6*AB.x*PCxxyym2zz*shALPHAb[bi]*shALPHAb[bi] - 12*rPC.x*(AB.x*rPC.x + AB.y*rPC.y - 2*AB.z*rPC.z)*gamma*eta)));
							factor+= (2*sqrtf(PC2gamma)*gamma*(PC*(-6*rPC.x*shALPHAb[bi] + 6*AB.x*PCxxyym2zz*shALPHAb[bi]*shALPHAb[bi] + 4*AB.x*PC2gamma*PCxxyym2zz*shALPHAb[bi]*shALPHAb[bi] + 4*AB.x*PC2gamma*eta + 4*ABxxyym2zz*PC2gamma*rPC.x*shALPHAa[ai]*eta - 8*AB.x*PC2gamma*(AB.x*rPC.x + AB.y*rPC.y - 2*AB.z*rPC.z)*shALPHAb[bi]*eta) + rPC.x*((15 + 10*PC2gamma + 4*PC2gamma2)*rPC.x*rPC.x*shALPHAb[bi] + (15 + 10*PC2gamma + 4*PC2gamma2)*rPC.y*rPC.y*shALPHAb[bi] - 4*AB.x*PC2gamma*(3 + 2*PC2gamma)*rPC.x*eta - 4*AB.y*PC2gamma*(3 + 2*PC2gamma)*rPC.y*eta - 2*rPC.z*((15 + 10*PC2gamma + 4*PC2gamma2)*rPC.z*shALPHAb[bi] - 4*AB.z*PC2gamma*(3 + 2*PC2gamma)*eta))))*expf(-PC2gamma);
							
							factor *= (PI*eta)/(8*PC2gamma3*sqrtf(PC2gamma)*shALPHAa[ai]*shALPHAb[bi]*shALPHAb[bi]*gamma);
							factor*= 0.488603*0.315392;

						}
						else if(shALMOa.z == 1 && shALMOb.z == 1) {

							factor = ERFgPC*PI_1over2*(15*rPC.x*rPC.x*rPC.z*shALPHAb[bi]*gamma + 4*AB.z*PC2gamma3*eta*(-1 + 2*AB.x*AB.x*eta) + PC2gamma2*(2*rPC.z*shALPHAb[bi] + 2*AB.z*eta + 4*AB.x*AB.z*rPC.x*shALPHAa[ai]*eta - 4*AB.x*(AB.z*rPC.x + AB.x*rPC.z)*shALPHAb[bi]*eta) + PC2gamma*(-3*rPC.z*shALPHAb[bi] + 6*AB.x*rPC.x*rPC.z*shALPHAb[bi]*shALPHAb[bi] - 6*rPC.x*(AB.z*rPC.x + AB.x*rPC.z)*gamma*eta));
							factor += (sqrtf(PC2gamma)*gamma*(PC*(-4*AB.z*PC2gamma*(1 + 2*AB.x*rPC.x*(shALPHAa[ai] - shALPHAb[bi]))*eta + 2*rPC.z*shALPHAb[bi]*(3 - 2*AB.x*(3 + 2*PC2gamma)*rPC.x*shALPHAb[bi] + 4*AB.x*AB.x*PC2gamma*eta)) - 2*rPC.x*(-2*AB.x*PC2gamma*(3 + 2*PC2gamma)*rPC.z*eta + rPC.x*((15 + 10*PC2gamma + 4*PC2gamma2)*rPC.z*shALPHAb[bi] - 2*AB.z*PC2gamma*(3 + 2*PC2gamma)*eta))))*expf(-PC2gamma);

							factor *= (PI*eta)/(8*PC2gamma3*sqrtf(PC2gamma)*shALPHAa[ai]*shALPHAb[bi]*shALPHAb[bi]*gamma);
							factor *= 0.488603*1.09255;

						}
						else if(shALMOa.z == 1 && shALMOb.z == 2) {

							factor = ERFgPC*PI_1over2*(15*(rPC.x*rPC.x*rPC.x - rPC.x*rPC.y*rPC.y)*shALPHAb[bi]*gamma + 8*AB.x*PC2gamma3*eta*(-1 + AB.x*AB.x*eta - AB.y*AB.y*eta) + 4*PC2gamma2*(rPC.x*shALPHAb[bi] + AB.x*eta + ABxxmyy*rPC.x*shALPHAa[ai]*eta + 2*AB.x*(-(AB.x*rPC.x) + AB.y*rPC.y)*shALPHAb[bi]*eta) + 6*PC2gamma*(-(rPC.x*shALPHAb[bi]) + AB.x*PCxxmyy*shALPHAb[bi]*shALPHAb[bi] + 2*rPC.x*(-(AB.x*rPC.x) + AB.y*rPC.y)*gamma*eta));
							factor+= (sqrtf(PC2gamma)*gamma*(-2*rPC.x*((15 + 10*PC2gamma + 4*PC2gamma2)*rPC.x*rPC.x*shALPHAb[bi] - 4*AB.x*PC2gamma*(3 + 2*PC2gamma)*rPC.x*eta + rPC.y*((-15 - 10*PC2gamma - 4*PC2gamma2)*rPC.y*shALPHAb[bi] + 4*AB.y*PC2gamma*(3 + 2*PC2gamma)*eta)) + 4*PC*(-(AB.x*(3 + 2*PC2gamma)*rPC.x*rPC.x*shALPHAb[bi]*shALPHAb[bi]) + AB.x*((3 + 2*PC2gamma)*rPC.y*rPC.y*shALPHAb[bi]*shALPHAb[bi] - 2*PC2gamma*eta - 4*AB.y*PC2gamma*rPC.y*shALPHAb[bi]*eta) + rPC.x*(-2*ABxxmyy*PC2gamma*shALPHAa[ai]*eta + shALPHAb[bi]*(3 + 4*AB.x*AB.x*PC2gamma*eta)))))*expf(-PC2gamma);

							factor *= (PI*eta)/(8*PC2gamma3*sqrtf(PC2gamma)*shALPHAa[ai]*shALPHAb[bi]*shALPHAb[bi]*gamma);
							factor *= 0.488603*0.546274;

						}

					} else if (shALMOa.y == 2 && shALMOb.y == 2) { // D-D of doom

						
						float PC2gamma2 = PC2gamma*PC2gamma;
						float PC2gamma3 = PC2gamma2*PC2gamma;
						float PCxxyym2zz = (rPC.x*rPC.x + rPC.y*rPC.y - 2*rPC.z*rPC.z);
						float ABxxyym2zz = (AB.x*AB.x + AB.y*AB.y - 2*AB.z*AB.z);
						//float PCxxmyy = rPC.x*rPC.x - rPC.y*rPC.y;
						float ABxxmyy = AB.x*AB.x - AB.y*AB.y;

						// THE DOOM OF DOOM

						if(shALMOa.z == -2 && shALMOb.z == -2) {

factor = -(ERFgPC*PI_1over2*(-15*PC2gamma*(rPC.y*rPC.y + rPC.x*rPC.x*(1 + 2*AB.y*rPC.y*(shALPHAa[ai] - shALPHAb[bi])) + 2*AB.x*rPC.x*rPC.y*rPC.y*(shALPHAa[ai] - shALPHAb[bi]))*gamma + 105*rPC.x*rPC.x*rPC.y*rPC.y*gamma*gamma + 4*PC2gamma2*PC2gamma2*(-1 + 2*AB.x*AB.x*eta)*(-1 + 2*AB.y*AB.y*eta) + 4*PC2gamma3*(-1 + AB.y*rPC.y*(-shALPHAa[ai] + shALPHAb[bi]) + AB.y*AB.y*eta + AB.x*AB.x*(1 + 2*AB.y*rPC.y*(shALPHAa[ai] - shALPHAb[bi]))*eta + AB.x*rPC.x*(shALPHAa[ai] - shALPHAb[bi])*(-1 + 2*AB.y*AB.y*eta)) - 3*PC2gamma2*(-1 + 2*AB.y*rPC.y*(-shALPHAa[ai] + shALPHAb[bi]) - 2*rPC.x*rPC.x*gamma - 2*rPC.y*rPC.y*gamma + 4*AB.y*AB.y*rPC.x*rPC.x*gamma*eta + 4*AB.x*AB.x*rPC.y*rPC.y*gamma*eta - 2*AB.x*rPC.x*(shALPHAa[ai] + 2*AB.y*rPC.y*shALPHAa[ai]*shALPHAa[ai] - shALPHAb[bi] + 2*AB.y*rPC.y*shALPHAb[bi]*shALPHAb[bi] - 4*AB.y*rPC.y*gamma*eta))));
factor+= (2*sqrtf(PC2gamma)*(105*rPC.x*rPC.x*rPC.y*rPC.y*gamma*gamma + 5*PC2gamma*gamma*(-3*rPC.y*rPC.y + 6*AB.x*rPC.x*rPC.y*rPC.y*(-shALPHAa[ai] + shALPHAb[bi]) + rPC.x*rPC.x*(-3 + 6*AB.y*rPC.y*(-shALPHAa[ai] + shALPHAb[bi]) + 14*rPC.y*rPC.y*gamma)) + PC2gamma2*(3 - 4*rPC.x*rPC.x*gamma - 4*rPC.y*rPC.y*gamma + 28*rPC.x*rPC.x*rPC.y*rPC.y*powf(gamma,2) - 2*AB.y*rPC.y*(shALPHAa[ai] - shALPHAb[bi])*(-3 + 10*rPC.x*rPC.x*gamma) - 12*AB.y*AB.y*rPC.x*rPC.x*gamma*eta - 12*AB.x*AB.x*rPC.y*rPC.y*gamma*eta + 2*AB.x*rPC.x*(6*AB.y*rPC.y*shALPHAa[ai]*shALPHAa[ai] - 3*shALPHAb[bi] + 6*AB.y*rPC.y*shALPHAb[bi]*shALPHAb[bi] + 10*rPC.y*rPC.y*shALPHAb[bi]*gamma + shALPHAa[ai]*(3 - 10*rPC.y*rPC.y*gamma) - 12*AB.y*rPC.y*gamma*eta)) + PC2gamma3*(-2 + 8*AB.y*rPC.x*rPC.x*rPC.y*(-shALPHAa[ai] + shALPHAb[bi])*gamma + 8*rPC.x*rPC.x*rPC.y*rPC.y*powf(gamma,2) + AB.y*AB.y*(4 - 8*rPC.x*rPC.x*gamma)*eta + 4*AB.x*AB.x*(1 + 2*AB.y*rPC.y*(shALPHAa[ai] - shALPHAb[bi]) - 2*rPC.y*rPC.y*gamma)*eta + 8*AB.x*rPC.x*(rPC.y*rPC.y*(-shALPHAa[ai] + shALPHAb[bi])*gamma + AB.y*AB.y*(shALPHAa[ai] - shALPHAb[bi])*eta + AB.y*rPC.y*(shALPHAa[ai]*shALPHAa[ai] + shALPHAb[bi]*shALPHAb[bi] - 2*gamma*eta)))))*expf(-PC2gamma);
							factor*= 1.09255*1.09255;
							factor *= PI/(16*sqrtf(PC2gamma)*PC2gamma2*PC2gamma2*gamma*gamma*gamma);

						}
						else if(shALMOa.z == -2 && shALMOb.z == -1) {

factor = -(ERFgPC*PI_1over2*(2*AB.z*PC2gamma*(-15*rPC.x*rPC.y*rPC.y*shALPHAa[ai]*gamma + 4*AB.x*PC2gamma3*eta*(-1 + 2*AB.y*AB.y*eta) + 3*PC2gamma*(rPC.x*shALPHAa[ai] + 2*AB.y*rPC.x*rPC.y*shALPHAa[ai]*shALPHAa[ai] - 2*rPC.y*(AB.y*rPC.x + AB.x*rPC.y)*gamma*eta) + 2*PC2gamma2*(AB.x*(1 + 2*AB.y*rPC.y*(shALPHAa[ai] - shALPHAb[bi]))*eta + rPC.x*shALPHAa[ai]*(-1 + 2*AB.y*AB.y*eta))) - rPC.z*(3*rPC.x*gamma*(5*PC2gamma*(1 + 2*AB.y*rPC.y*(shALPHAa[ai] - shALPHAb[bi])) - 35*rPC.y*rPC.y*gamma + PC2gamma2*(-2 + 4*AB.y*AB.y*eta)) + 2*AB.x*PC2gamma*(-15*rPC.y*rPC.y*shALPHAb[bi]*gamma + 2*PC2gamma2*shALPHAb[bi]*(-1 + 2*AB.y*AB.y*eta) + PC2gamma*(3*shALPHAb[bi] - 6*AB.y*rPC.y*shALPHAb[bi]*shALPHAb[bi] + 6*AB.y*rPC.y*gamma*eta)))));
factor+= (2*sqrtf(PC2gamma)*gamma*(105*rPC.x*rPC.y*rPC.y*rPC.z*gamma + PC2gamma*(6*AB.x*rPC.z*shALPHAb[bi]*(5*rPC.y*rPC.y + PC*(-1 + 2*AB.y*rPC.y*shALPHAb[bi])) + rPC.x*(6*AB.z*shALPHAa[ai]*(-5*rPC.y*rPC.y + PC*(1 + 2*AB.y*rPC.y*shALPHAa[ai])) - 5*rPC.z*(3 + 6*AB.y*rPC.y*(shALPHAa[ai] - shALPHAb[bi]) - 14*rPC.y*rPC.y*gamma))) + 4*PC2gamma2*(-(rPC.x*rPC.z) - 5*rPC.x*rPC.y*(AB.z*rPC.y + AB.y*rPC.z)*shALPHAa[ai] + 2*AB.y*AB.z*PC*rPC.x*rPC.y*shALPHAa[ai]*shALPHAa[ai] + 5*rPC.y*(AB.y*rPC.x + AB.x*rPC.y)*rPC.z*shALPHAb[bi] + 2*AB.x*AB.y*PC*rPC.y*rPC.z*shALPHAb[bi]*shALPHAb[bi] + 7*rPC.x*rPC.y*rPC.y*rPC.z*gamma + AB.x*AB.z*PC*eta - 3*(AB.y*rPC.x + AB.x*rPC.y)*(AB.z*rPC.y + AB.y*rPC.z)*eta + 2*AB.y*AB.z*PC*(AB.y*rPC.x + AB.x*rPC.y)*shALPHAa[ai]*eta - 2*AB.x*AB.y*PC*(AB.z*rPC.y + AB.y*rPC.z)*shALPHAb[bi]*eta) - 8*PC2gamma3*(AB.z*rPC.y*(rPC.x*rPC.y*shALPHAa[ai] + AB.y*rPC.x*eta + AB.x*rPC.y*eta) + rPC.z*(-(rPC.y*rPC.y*(AB.x*shALPHAb[bi] + rPC.x*gamma)) + AB.y*AB.y*rPC.x*eta + AB.y*rPC.y*(rPC.x*shALPHAa[ai] - rPC.x*shALPHAb[bi] + AB.x*eta)))))*expf(-PC2gamma);
							factor*= 1.09255*1.09255;
							factor *= PI/(16*sqrtf(PC2gamma)*PC2gamma2*PC2gamma2*gamma*gamma*gamma);

						}
						else if(shALMOa.z == -2 && shALMOb.z == 0) {

factor = ERFgPC*PI_1over2*(-8*AB.y*AB.y*PC2gamma2*rPC.y*(2*AB.x*PC2gamma*shALPHAb[bi] + 3*rPC.x*gamma)*eta + rPC.y*(3*rPC.x*(4*ABxxyym2zz*PC2gamma2*shALPHAa[ai]*shALPHAa[ai] + gamma*(-20*PC2gamma + 8*PC2gamma2 + 40*AB.z*PC2gamma*rPC.z*shALPHAa[ai] + 35*PCxxyym2zz*gamma)) - 24*AB.x*AB.x*PC2gamma2*rPC.x*gamma*eta + 2*AB.x*PC2gamma*(15*(-2*rPC.x*rPC.x*shALPHAa[ai] + PCxxyym2zz*shALPHAb[bi])*gamma + 4*PC2gamma2*(-shALPHAa[ai] + shALPHAb[bi] + ABxxyym2zz*shALPHAa[ai]*eta) + 6*PC2gamma*(shALPHAa[ai] - shALPHAb[bi] + 4*AB.z*rPC.z*gamma*eta))) + 2*AB.y*PC2gamma*(15*rPC.x*(-2*rPC.y*rPC.y*shALPHAa[ai] + PCxxyym2zz*shALPHAb[bi])*gamma + 8*AB.x*PC2gamma3*eta*(-2 + ABxxyym2zz*eta) + 4*PC2gamma2*(2*AB.x*(1 + 2*AB.z*rPC.z*shALPHAb[bi])*eta + rPC.x*(-shALPHAa[ai] + shALPHAb[bi] + ABxxyym2zz*shALPHAa[ai]*eta - 2*AB.x*AB.x*shALPHAb[bi]*eta)) + 6*PC2gamma*(rPC.x*(shALPHAa[ai] - shALPHAb[bi] + 4*AB.z*rPC.z*gamma*eta) + AB.x*(PCxxyym2zz*shALPHAb[bi]*shALPHAb[bi] - 2*PCxxyym2zz*gamma*eta - 4*rPC.z*rPC.z*gamma*eta))));
factor+= (2*sqrtf(PC2gamma)*gamma*(-105*rPC.x*PCxxyym2zz*rPC.y*gamma - 2*PC2gamma*(3*AB.x*(5*rPC.y*(-2*PCxxyym2zz*shALPHAa[ai] + 2*rPC.y*rPC.y*shALPHAa[ai] - 4*rPC.z*rPC.z*shALPHAa[ai] + PCxxyym2zz*shALPHAb[bi]) + 2*PC*(rPC.y*shALPHAa[ai] - rPC.y*shALPHAb[bi] + AB.y*PCxxyym2zz*shALPHAb[bi]*shALPHAb[bi])) + rPC.x*(-30*AB.y*rPC.y*rPC.y*shALPHAa[ai] + 3*AB.y*(2*PC*(shALPHAa[ai] - shALPHAb[bi]) + 5*PCxxyym2zz*shALPHAb[bi]) + rPC.y*(-30 + 60*AB.z*rPC.z*shALPHAa[ai] + 6*ABxxyym2zz*PC*shALPHAa[ai]*shALPHAa[ai] + 35*PCxxyym2zz*gamma))) + 8*PC2gamma3*(2*rPC.x*rPC.y*(AB.x*rPC.x + AB.y*rPC.y - 2*AB.z*rPC.z)*shALPHAa[ai] - PCxxyym2zz*(AB.y*rPC.x + AB.x*rPC.y)*shALPHAb[bi] - rPC.x*PCxxyym2zz*rPC.y*gamma + 2*(AB.y*rPC.x + AB.x*rPC.y)*(AB.x*rPC.x + AB.y*rPC.y - 2*AB.z*rPC.z)*eta) + 4*PC2gamma2*(4*rPC.x*rPC.y + 10*rPC.x*rPC.y*(AB.x*rPC.x + AB.y*rPC.y - 2*AB.z*rPC.z)*shALPHAa[ai] - 2*ABxxyym2zz*PC*rPC.x*rPC.y*shALPHAa[ai]*shALPHAa[ai] - 5*PCxxyym2zz*(AB.y*rPC.x + AB.x*rPC.y)*shALPHAb[bi] - 2*AB.x*AB.y*PC*PCxxyym2zz*shALPHAb[bi]*shALPHAb[bi] - 7*rPC.x*PCxxyym2zz*rPC.y*gamma - 4*AB.x*AB.y*PC*eta + 6*(AB.y*rPC.x + AB.x*rPC.y)*(AB.x*rPC.x + AB.y*rPC.y - 2*AB.z*rPC.z)*eta - 2*ABxxyym2zz*PC*(AB.y*rPC.x + AB.x*rPC.y)*shALPHAa[ai]*eta + 4*AB.x*AB.y*PC*(AB.x*rPC.x + AB.y*rPC.y - 2*AB.z*rPC.z)*shALPHAb[bi]*eta)))*expf(-PC2gamma);

							factor*= 1.09255*0.315392;
							factor *= PI/(16*sqrtf(PC2gamma)*PC2gamma2*PC2gamma2*gamma*gamma*gamma);

						}
						else if(shALMOa.z == -2 && shALMOb.z == 1) {

factor = -(ERFgPC*PI_1over2*(2*AB.z*PC2gamma*(-15*rPC.x*rPC.x*rPC.y*shALPHAa[ai]*gamma + 4*AB.y*PC2gamma3*eta*(-1 + 2*AB.x*AB.x*eta) + 3*PC2gamma*(rPC.y*shALPHAa[ai] + 2*AB.x*rPC.x*rPC.y*shALPHAa[ai]*shALPHAa[ai] - 2*rPC.x*(AB.y*rPC.x + AB.x*rPC.y)*gamma*eta) + 2*PC2gamma2*(AB.y*(1 + 2*AB.x*rPC.x*(shALPHAa[ai] - shALPHAb[bi]))*eta + rPC.y*shALPHAa[ai]*(-1 + 2*AB.x*AB.x*eta))) - rPC.z*(3*rPC.y*gamma*(5*PC2gamma*(1 + 2*AB.x*rPC.x*(shALPHAa[ai] - shALPHAb[bi])) - 35*rPC.x*rPC.x*gamma + PC2gamma2*(-2 + 4*AB.x*AB.x*eta)) + 2*AB.y*PC2gamma*(-15*rPC.x*rPC.x*shALPHAb[bi]*gamma + 2*PC2gamma2*shALPHAb[bi]*(-1 + 2*AB.x*AB.x*eta) + PC2gamma*(3*shALPHAb[bi] - 6*AB.x*rPC.x*shALPHAb[bi]*shALPHAb[bi] + 6*AB.x*rPC.x*gamma*eta)))));
factor+= (2*sqrtf(PC2gamma)*gamma*(105*rPC.x*rPC.x*rPC.y*rPC.z*gamma + PC2gamma*(6*AB.y*rPC.z*shALPHAb[bi]*(5*rPC.x*rPC.x + PC*(-1 + 2*AB.x*rPC.x*shALPHAb[bi])) + rPC.y*(6*AB.z*shALPHAa[ai]*(-5*rPC.x*rPC.x + PC*(1 + 2*AB.x*rPC.x*shALPHAa[ai])) - 5*rPC.z*(3 + 6*AB.x*rPC.x*(shALPHAa[ai] - shALPHAb[bi]) - 14*rPC.x*rPC.x*gamma))) + 4*PC2gamma2*(-(rPC.y*rPC.z) - 5*rPC.x*rPC.y*(AB.z*rPC.x + AB.x*rPC.z)*shALPHAa[ai] + 2*AB.x*AB.z*PC*rPC.x*rPC.y*shALPHAa[ai]*shALPHAa[ai] + 5*rPC.x*(AB.y*rPC.x + AB.x*rPC.y)*rPC.z*shALPHAb[bi] + 2*AB.x*AB.y*PC*rPC.x*rPC.z*shALPHAb[bi]*shALPHAb[bi] + 7*rPC.x*rPC.x*rPC.y*rPC.z*gamma + AB.y*AB.z*PC*eta - 3*(AB.y*rPC.x + AB.x*rPC.y)*(AB.z*rPC.x + AB.x*rPC.z)*eta + 2*AB.x*AB.z*PC*(AB.y*rPC.x + AB.x*rPC.y)*shALPHAa[ai]*eta - 2*AB.x*AB.y*PC*(AB.z*rPC.x + AB.x*rPC.z)*shALPHAb[bi]*eta) - 8*PC2gamma3*(AB.z*rPC.x*(rPC.x*rPC.y*shALPHAa[ai] + AB.y*rPC.x*eta + AB.x*rPC.y*eta) + rPC.z*(-(rPC.x*rPC.x*(AB.y*shALPHAb[bi] + rPC.y*gamma)) + AB.x*AB.x*rPC.y*eta + AB.x*rPC.x*(rPC.y*shALPHAa[ai] - rPC.y*shALPHAb[bi] + AB.y*eta)))))*expf(-PC2gamma);
							
							factor*= 1.09255*1.09255;
							factor *= PI/(16*sqrtf(PC2gamma)*PC2gamma2*PC2gamma2*gamma*gamma*gamma);

						}
						else if(shALMOa.z == -2 && shALMOb.z == 2) {

factor = ERFgPC*PI_1over2*(8*powf(AB.y,3)*PC2gamma3*eta*(rPC.x*shALPHAa[ai] + 2*AB.x*PC2gamma*eta) + 4*AB.y*AB.y*PC2gamma2*rPC.y*(2*AB.x*PC2gamma*(shALPHAa[ai] - 2*shALPHAb[bi])*eta + 3*rPC.x*(shALPHAa[ai]*shALPHAa[ai] - 2*gamma*eta)) + rPC.y*(105*rPC.x*(-rPC.x*rPC.x + rPC.y*rPC.y)*powf(gamma,2) + 2*AB.x*PC2gamma*(-6*PC2gamma*(shALPHAa[ai] + shALPHAb[bi]) + 4*PC2gamma2*(shALPHAa[ai] + shALPHAb[bi]) + 15*(rPC.x*rPC.x*(2*shALPHAa[ai] - shALPHAb[bi]) + rPC.y*rPC.y*shALPHAb[bi])*gamma) - 8*powf(AB.x,3)*PC2gamma3*shALPHAa[ai]*eta - 12*AB.x*AB.x*PC2gamma2*rPC.x*(shALPHAa[ai]*shALPHAa[ai] - 2*gamma*eta)) - 2*AB.y*PC2gamma*(15*rPC.x*(rPC.y*rPC.y*(2*shALPHAa[ai] - shALPHAb[bi]) + rPC.x*rPC.x*shALPHAb[bi])*gamma + 8*powf(AB.x,3)*PC2gamma3*eta*eta + 4*PC2gamma2*rPC.x*(shALPHAa[ai] + shALPHAb[bi] + AB.x*AB.x*shALPHAa[ai]*eta - 2*AB.x*AB.x*shALPHAb[bi]*eta) + 6*PC2gamma*(-(rPC.x*(shALPHAa[ai] + shALPHAb[bi])) + AB.x*rPC.x*rPC.x*(shALPHAb[bi]*shALPHAb[bi] - 2*gamma*eta) - AB.x*rPC.y*rPC.y*(shALPHAb[bi]*shALPHAb[bi] - 2*gamma*eta))));
factor+= (2*sqrtf(PC2gamma)*gamma*(-8*powf(AB.y,3)*PC*PC2gamma2*rPC.x*shALPHAa[ai]*eta + rPC.y*(2*AB.x*PC2gamma*(6*PC*(shALPHAa[ai] + shALPHAb[bi]) - (15 + 10*PC2gamma + 4*PC2gamma2)*(rPC.x*rPC.x*(2*shALPHAa[ai] - shALPHAb[bi]) + rPC.y*rPC.y*shALPHAb[bi])) + (105 + 70*PC2gamma + 28*PC2gamma2 + 8*PC2gamma3)*rPC.x*(rPC.x*rPC.x - rPC.y*rPC.y)*gamma + 8*powf(AB.x,3)*PC*PC2gamma2*shALPHAa[ai]*eta + 4*AB.x*AB.x*PC2gamma*(3 + 2*PC2gamma)*rPC.x*(PC*shALPHAa[ai]*shALPHAa[ai] - 2*PC2gamma*eta)) - 4*AB.y*AB.y*PC2gamma*rPC.y*(-2*PC2gamma*(3 + 2*PC2gamma)*rPC.x*eta + PC*((3 + 2*PC2gamma)*rPC.x*shALPHAa[ai]*shALPHAa[ai] + 2*AB.x*PC2gamma*(shALPHAa[ai] - 2*shALPHAb[bi])*eta)) + 2*AB.y*PC2gamma*((15 + 10*PC2gamma + 4*PC2gamma2)*rPC.x*rPC.y*rPC.y*(2*shALPHAa[ai] - shALPHAb[bi]) + (15 + 10*PC2gamma + 4*PC2gamma2)*powf(rPC.x,3)*shALPHAb[bi] - 4*AB.x*PC2gamma*(3 + 2*PC2gamma)*rPC.x*rPC.x*eta + 4*AB.x*PC2gamma*(3 + 2*PC2gamma)*rPC.y*rPC.y*eta + PC*(2*AB.x*(3 + 2*PC2gamma)*rPC.x*rPC.x*shALPHAb[bi]*shALPHAb[bi] - 2*AB.x*(3 + 2*PC2gamma)*rPC.y*rPC.y*shALPHAb[bi]*shALPHAb[bi] + rPC.x*(-6*shALPHAa[ai] - 6*shALPHAb[bi] + 4*AB.x*AB.x*PC2gamma*shALPHAa[ai]*eta - 8*AB.x*AB.x*PC2gamma*shALPHAb[bi]*eta)))))*expf(-PC2gamma);

							factor*= 1.09255*0.546274;
							factor *= PI/(16*sqrtf(PC2gamma)*PC2gamma2*PC2gamma2*gamma*gamma*gamma);

						}
						else if(shALMOa.z == -1 && shALMOb.z == -1) {

factor = -(ERFgPC*PI_1over2*(-15*PC2gamma*(rPC.z*rPC.z + rPC.y*rPC.y*(1 + 2*AB.z*rPC.z*(shALPHAa[ai] - shALPHAb[bi])) + 2*AB.y*rPC.y*rPC.z*rPC.z*(shALPHAa[ai] - shALPHAb[bi]))*gamma + 105*rPC.y*rPC.y*rPC.z*rPC.z*powf(gamma,2) + 4*PC2gamma2*PC2gamma2*(-1 + 2*AB.y*AB.y*eta)*(-1 + 2*AB.z*AB.z*eta) + 4*PC2gamma3*(-1 + AB.z*rPC.z*(-shALPHAa[ai] + shALPHAb[bi]) + AB.z*AB.z*eta + AB.y*AB.y*(1 + 2*AB.z*rPC.z*(shALPHAa[ai] - shALPHAb[bi]))*eta + AB.y*rPC.y*(shALPHAa[ai] - shALPHAb[bi])*(-1 + 2*AB.z*AB.z*eta)) - 3*PC2gamma2*(-1 + 2*AB.z*rPC.z*(-shALPHAa[ai] + shALPHAb[bi]) - 2*rPC.y*rPC.y*gamma - 2*rPC.z*rPC.z*gamma + 4*AB.z*AB.z*rPC.y*rPC.y*gamma*eta + 4*AB.y*AB.y*rPC.z*rPC.z*gamma*eta - 2*AB.y*rPC.y*(shALPHAa[ai] + 2*AB.z*rPC.z*shALPHAa[ai]*shALPHAa[ai] - shALPHAb[bi] + 2*AB.z*rPC.z*shALPHAb[bi]*shALPHAb[bi] - 4*AB.z*rPC.z*gamma*eta))));
factor+= (2*sqrtf(PC2gamma)*(105*rPC.y*rPC.y*rPC.z*rPC.z*powf(gamma,2) + 5*PC2gamma*gamma*(-3*rPC.z*rPC.z + 6*AB.y*rPC.y*rPC.z*rPC.z*(-shALPHAa[ai] + shALPHAb[bi]) + rPC.y*rPC.y*(-3 + 6*AB.z*rPC.z*(-shALPHAa[ai] + shALPHAb[bi]) + 14*rPC.z*rPC.z*gamma)) + PC2gamma2*(3 - 4*rPC.y*rPC.y*gamma - 4*rPC.z*rPC.z*gamma + 28*rPC.y*rPC.y*rPC.z*rPC.z*powf(gamma,2) - 2*AB.z*rPC.z*(shALPHAa[ai] - shALPHAb[bi])*(-3 + 10*rPC.y*rPC.y*gamma) - 12*AB.z*AB.z*rPC.y*rPC.y*gamma*eta - 12*AB.y*AB.y*rPC.z*rPC.z*gamma*eta + 2*AB.y*rPC.y*(6*AB.z*rPC.z*shALPHAa[ai]*shALPHAa[ai] - 3*shALPHAb[bi] + 6*AB.z*rPC.z*shALPHAb[bi]*shALPHAb[bi] + 10*rPC.z*rPC.z*shALPHAb[bi]*gamma + shALPHAa[ai]*(3 - 10*rPC.z*rPC.z*gamma) - 12*AB.z*rPC.z*gamma*eta)) + PC2gamma3*(-2 + 8*AB.z*rPC.y*rPC.y*rPC.z*(-shALPHAa[ai] + shALPHAb[bi])*gamma + 8*rPC.y*rPC.y*rPC.z*rPC.z*powf(gamma,2) + AB.z*AB.z*(4 - 8*rPC.y*rPC.y*gamma)*eta + 4*AB.y*AB.y*(1 + 2*AB.z*rPC.z*(shALPHAa[ai] - shALPHAb[bi]) - 2*rPC.z*rPC.z*gamma)*eta + 8*AB.y*rPC.y*(rPC.z*rPC.z*(-shALPHAa[ai] + shALPHAb[bi])*gamma + AB.z*AB.z*(shALPHAa[ai] - shALPHAb[bi])*eta + AB.z*rPC.z*(shALPHAa[ai]*shALPHAa[ai] + shALPHAb[bi]*shALPHAb[bi] - 2*gamma*eta)))))*expf(-PC2gamma);

							factor*= 1.09255*1.09255;
							factor *= PI/(16*sqrtf(PC2gamma)*PC2gamma2*PC2gamma2*gamma*gamma*gamma);

						}
						else if(shALMOa.z == -1 && shALMOb.z == 0) {

factor = ERFgPC*PI_1over2*(-16*AB.z*AB.z*PC2gamma2*rPC.z*(2*AB.y*PC2gamma*shALPHAb[bi] + 3*rPC.y*gamma)*eta + rPC.z*(-3*rPC.y*(4*ABxxyym2zz*PC2gamma2*shALPHAa[ai]*shALPHAa[ai] + gamma*(10*PC2gamma - 4*PC2gamma2 - 20*AB.x*PC2gamma*rPC.x*shALPHAa[ai] + 35*PCxxyym2zz*gamma)) + 24*AB.y*AB.y*PC2gamma2*rPC.y*gamma*eta - 2*AB.y*PC2gamma*(15*(-2*rPC.y*rPC.y*shALPHAa[ai] + PCxxyym2zz*shALPHAb[bi])*gamma - 4*PC2gamma2*(shALPHAa[ai] + 2*shALPHAb[bi] - ABxxyym2zz*shALPHAa[ai]*eta) + 6*PC2gamma*(shALPHAa[ai] + 2*shALPHAb[bi] - 2*AB.x*rPC.x*gamma*eta))) - 2*AB.z*PC2gamma*(15*rPC.y*(4*rPC.z*rPC.z*shALPHAa[ai] + PCxxyym2zz*shALPHAb[bi])*gamma + 8*AB.y*PC2gamma3*eta*(1 + ABxxyym2zz*eta) + 4*PC2gamma2*(-(AB.y*(1 + 2*AB.x*rPC.x*shALPHAb[bi])*eta) + rPC.y*(2*shALPHAa[ai] + shALPHAb[bi] + ABxxyym2zz*shALPHAa[ai]*eta - 2*AB.y*AB.y*shALPHAb[bi]*eta)) - 6*PC2gamma*(2*AB.y*rPC.y*rPC.y*gamma*eta + rPC.y*(2*shALPHAa[ai] + shALPHAb[bi] + 2*AB.x*rPC.x*gamma*eta) - AB.y*(PCxxyym2zz*shALPHAb[bi]*shALPHAb[bi] + 4*rPC.z*rPC.z*gamma*eta))));
factor+= (sqrtf(PC2gamma)*gamma*(210*PCxxyym2zz*rPC.y*rPC.z*gamma + 4*PC2gamma*(-30*AB.y*rPC.y*rPC.y*rPC.z*shALPHAa[ai] + 3*AB.y*(5*PCxxyym2zz*rPC.z*shALPHAb[bi] + 2*PC*(rPC.z*shALPHAa[ai] + 2*rPC.z*shALPHAb[bi] + AB.z*PCxxyym2zz*shALPHAb[bi]*shALPHAb[bi])) + rPC.y*(60*AB.z*rPC.z*rPC.z*shALPHAa[ai] - 3*AB.z*(-5*PCxxyym2zz*shALPHAb[bi] + 2*PC*(2*shALPHAa[ai] + shALPHAb[bi])) + rPC.z*(15 - 30*AB.x*rPC.x*shALPHAa[ai] + 6*ABxxyym2zz*PC*shALPHAa[ai]*shALPHAa[ai] + 35*PCxxyym2zz*gamma))) - 16*PC2gamma3*(2*rPC.y*rPC.z*(AB.x*rPC.x + AB.y*rPC.y - 2*AB.z*rPC.z)*shALPHAa[ai] - PCxxyym2zz*(AB.z*rPC.y + AB.y*rPC.z)*shALPHAb[bi] - PCxxyym2zz*rPC.y*rPC.z*gamma + 2*(AB.z*rPC.y + AB.y*rPC.z)*(AB.x*rPC.x + AB.y*rPC.y - 2*AB.z*rPC.z)*eta) + 8*PC2gamma2*(-2*AB.y*rPC.y*rPC.y*(5*rPC.z*shALPHAa[ai] + 3*AB.z*eta) + AB.y*(PCxxyym2zz*shALPHAb[bi]*(5*rPC.z + 2*AB.z*PC*shALPHAb[bi]) + 2*(-3*AB.x*rPC.x*rPC.z + ABxxyym2zz*PC*rPC.z*shALPHAa[ai] + 4*AB.z*AB.z*PC*rPC.z*shALPHAb[bi] - AB.z*(PC - 6*rPC.z*rPC.z + 2*AB.x*PC*rPC.x*shALPHAb[bi]))*eta) + rPC.y*(20*AB.z*rPC.z*rPC.z*shALPHAa[ai] + rPC.z*(2 - 10*AB.x*rPC.x*shALPHAa[ai] + 2*ABxxyym2zz*PC*shALPHAa[ai]*shALPHAa[ai] + 7*PCxxyym2zz*gamma - 6*AB.y*AB.y*eta + 12*AB.z*AB.z*eta) + AB.z*(5*PCxxyym2zz*shALPHAb[bi] + 2*(-3*AB.x*rPC.x + PC*(ABxxyym2zz*shALPHAa[ai] - 2*AB.y*AB.y*shALPHAb[bi]))*eta)))))*expf(-PC2gamma);

							factor*= 1.09255*0.315392;
							factor *= -PI/(16*sqrtf(PC2gamma)*PC2gamma2*PC2gamma2*gamma*gamma*gamma);

						}
						else if(shALMOa.z == -1 && shALMOb.z == 1) {

factor = ERFgPC*PI_1over2*(2*AB.x*PC2gamma*(-15*rPC.y*rPC.z*rPC.z*shALPHAa[ai]*gamma + 4*AB.y*PC2gamma3*eta*(-1 + 2*AB.z*AB.z*eta) + 3*PC2gamma*(rPC.y*shALPHAa[ai] + 2*AB.z*rPC.y*rPC.z*shALPHAa[ai]*shALPHAa[ai] - 2*rPC.z*(AB.z*rPC.y + AB.y*rPC.z)*gamma*eta) + 2*PC2gamma2*(AB.y*(1 + 2*AB.z*rPC.z*(shALPHAa[ai] - shALPHAb[bi]))*eta + rPC.y*shALPHAa[ai]*(-1 + 2*AB.z*AB.z*eta))) - rPC.x*(3*rPC.y*gamma*(5*PC2gamma*(1 + 2*AB.z*rPC.z*(shALPHAa[ai] - shALPHAb[bi])) - 35*rPC.z*rPC.z*gamma + PC2gamma2*(-2 + 4*AB.z*AB.z*eta)) + 2*AB.y*PC2gamma*(-15*rPC.z*rPC.z*shALPHAb[bi]*gamma + 2*PC2gamma2*shALPHAb[bi]*(-1 + 2*AB.z*AB.z*eta) + PC2gamma*(3*shALPHAb[bi] - 6*AB.z*rPC.z*shALPHAb[bi]*shALPHAb[bi] + 6*AB.z*rPC.z*gamma*eta))));
factor+= (2*sqrtf(PC2gamma)*gamma*(105*rPC.x*rPC.y*rPC.z*rPC.z*gamma + PC2gamma*(6*AB.x*rPC.y*shALPHAa[ai]*(-5*rPC.z*rPC.z + PC*(1 + 2*AB.z*rPC.z*shALPHAa[ai])) + rPC.x*(6*AB.y*shALPHAb[bi]*(5*rPC.z*rPC.z + PC*(-1 + 2*AB.z*rPC.z*shALPHAb[bi])) - 5*rPC.y*(3 + 6*AB.z*rPC.z*(shALPHAa[ai] - shALPHAb[bi]) - 14*rPC.z*rPC.z*gamma))) - 4*PC2gamma2*(rPC.x*rPC.y + 5*rPC.y*rPC.z*(AB.z*rPC.x + AB.x*rPC.z)*shALPHAa[ai] - 2*AB.x*AB.z*PC*rPC.y*rPC.z*shALPHAa[ai]*shALPHAa[ai] - 5*rPC.x*rPC.z*(AB.z*rPC.y + AB.y*rPC.z)*shALPHAb[bi] - 2*AB.y*AB.z*PC*rPC.x*rPC.z*shALPHAb[bi]*shALPHAb[bi] - 7*rPC.x*rPC.y*rPC.z*rPC.z*gamma - AB.x*AB.y*PC*eta + 3*(AB.z*rPC.x + AB.x*rPC.z)*(AB.z*rPC.y + AB.y*rPC.z)*eta - 2*AB.x*AB.z*PC*(AB.z*rPC.y + AB.y*rPC.z)*shALPHAa[ai]*eta + 2*AB.y*AB.z*PC*(AB.z*rPC.x + AB.x*rPC.z)*shALPHAb[bi]*eta) - 8*PC2gamma3*(AB.z*AB.z*rPC.x*rPC.y*eta + rPC.z*rPC.z*(AB.x*rPC.y*shALPHAa[ai] - AB.y*rPC.x*shALPHAb[bi] - rPC.x*rPC.y*gamma + AB.x*AB.y*eta) + AB.z*rPC.z*(rPC.x*rPC.y*shALPHAa[ai] - rPC.x*rPC.y*shALPHAb[bi] + AB.y*rPC.x*eta + AB.x*rPC.y*eta))))*expf(-PC2gamma);

							factor*= 1.09255*1.09255;
							factor *= PI/(16*sqrtf(PC2gamma)*PC2gamma2*PC2gamma2*gamma*gamma*gamma);

						}
						else if(shALMOa.z == -1 && shALMOb.z == 2) {

factor = ERFgPC*PI_1over2*(12*AB.y*PC2gamma2*rPC.z*shALPHAa[ai] - 8*AB.y*PC2gamma3*rPC.z*shALPHAa[ai] + 12*(-AB.x*AB.x + AB.y*AB.y)*PC2gamma2*rPC.y*rPC.z*shALPHAa[ai]*shALPHAa[ai] - 12*AB.z*PC2gamma2*rPC.y*shALPHAb[bi] + 8*AB.z*PC2gamma3*rPC.y*shALPHAb[bi] + 12*AB.y*AB.z*PC2gamma2*(-rPC.x*rPC.x + rPC.y*rPC.y)*shALPHAb[bi]*shALPHAb[bi] - 30*PC2gamma*rPC.y*rPC.z*gamma + 12*PC2gamma2*rPC.y*rPC.z*gamma + 60*PC2gamma*rPC.y*(AB.x*rPC.x - AB.y*rPC.y)*rPC.z*shALPHAa[ai]*gamma - 30*PC2gamma*(rPC.x*rPC.x - rPC.y*rPC.y)*(AB.z*rPC.y + AB.y*rPC.z)*shALPHAb[bi]*gamma + 105*rPC.y*(-rPC.x*rPC.x + rPC.y*rPC.y)*rPC.z*powf(gamma,2) + 8*AB.y*AB.z*PC2gamma3*eta - 16*AB.y*AB.z*PC2gamma2*PC2gamma2*eta - 8*(AB.x*AB.x - AB.y*AB.y)*PC2gamma3*(AB.z*rPC.y + AB.y*rPC.z)*shALPHAa[ai]*eta + 16*AB.y*AB.z*PC2gamma3*(AB.x*rPC.x - AB.y*rPC.y)*shALPHAb[bi]*eta + 24*PC2gamma2*(AB.x*rPC.x - AB.y*rPC.y)*(AB.z*rPC.y + AB.y*rPC.z)*gamma*eta + 16*AB.y*(-AB.x*AB.x + AB.y*AB.y)*AB.z*PC2gamma2*PC2gamma2*eta*eta);
factor+= (2*sqrtf(PC2gamma)*gamma*(105*rPC.y*(rPC.x*rPC.x - rPC.y*rPC.y)*rPC.z*gamma + 2*PC2gamma*(15*rPC.y*rPC.z - 6*AB.y*PC*rPC.z*shALPHAa[ai] + 30*rPC.y*(-(AB.x*rPC.x) + AB.y*rPC.y)*rPC.z*shALPHAa[ai] + 6*(AB.x*AB.x - AB.y*AB.y)*PC*rPC.y*rPC.z*shALPHAa[ai]*shALPHAa[ai] + 6*AB.z*PC*rPC.y*shALPHAb[bi] + 15*(rPC.x*rPC.x - rPC.y*rPC.y)*(AB.z*rPC.y + AB.y*rPC.z)*shALPHAb[bi] + 6*AB.y*AB.z*PC*(rPC.x*rPC.x - rPC.y*rPC.y)*shALPHAb[bi]*shALPHAb[bi] + 35*rPC.y*(rPC.x*rPC.x - rPC.y*rPC.y)*rPC.z*gamma) - 8*PC2gamma3*(2*rPC.y*(AB.x*rPC.x - AB.y*rPC.y)*rPC.z*shALPHAa[ai] - (rPC.x*rPC.x - rPC.y*rPC.y)*(AB.z*rPC.y + AB.y*rPC.z)*shALPHAb[bi] + rPC.y*(-rPC.x*rPC.x + rPC.y*rPC.y)*rPC.z*gamma + 2*(AB.x*rPC.x - AB.y*rPC.y)*(AB.z*rPC.y + AB.y*rPC.z)*eta) + 4*PC2gamma2*(2*rPC.y*rPC.z + 10*rPC.y*(-(AB.x*rPC.x) + AB.y*rPC.y)*rPC.z*shALPHAa[ai] + 2*(AB.x*AB.x - AB.y*AB.y)*PC*rPC.y*rPC.z*shALPHAa[ai]*shALPHAa[ai] + 5*(rPC.x*rPC.x - rPC.y*rPC.y)*(AB.z*rPC.y + AB.y*rPC.z)*shALPHAb[bi] + 2*AB.y*AB.z*PC*(rPC.x*rPC.x - rPC.y*rPC.y)*shALPHAb[bi]*shALPHAb[bi] + 7*rPC.y*(rPC.x*rPC.x - rPC.y*rPC.y)*rPC.z*gamma - 2*AB.y*AB.z*PC*eta - 6*(AB.x*rPC.x - AB.y*rPC.y)*(AB.z*rPC.y + AB.y*rPC.z)*eta + 2*(AB.x*AB.x - AB.y*AB.y)*PC*(AB.z*rPC.y + AB.y*rPC.z)*shALPHAa[ai]*eta + 4*AB.y*AB.z*PC*(-(AB.x*rPC.x) + AB.y*rPC.y)*shALPHAb[bi]*eta)))*expf(-PC2gamma);

							factor*= 1.09255*0.546274;
							factor *= PI/(16*sqrtf(PC2gamma)*PC2gamma2*PC2gamma2*gamma*gamma*gamma);

						}
						else if(shALMOa.z == 0 && shALMOb.z == 0) {

factor = ERFgPC*sqrtf(PC)*PI_1over2*(-60*PC2gamma*(-6*rPC.z*rPC.z + PCxxyym2zz*(-1 + 2*AB.z*rPC.z*shALPHAa[ai] - 2*AB.z*rPC.z*shALPHAb[bi] + AB.x*rPC.x*(-shALPHAa[ai] + shALPHAb[bi]) + AB.y*rPC.y*(-shALPHAa[ai] + shALPHAb[bi])))*gamma - 105*powf(PCxxyym2zz,2)*powf(gamma,2) - 16*PC2gamma2*PC2gamma2*(3 - 2*ABxxyym2zz*eta - 12*AB.z*AB.z*eta + powf(ABxxyym2zz,2)*eta*eta) - 16*PC2gamma3*(-3 - 4*AB.z*rPC.z*shALPHAa[ai] + 4*AB.z*rPC.z*shALPHAb[bi] + ABxxyym2zz*eta + 6*AB.z*AB.z*eta - 2*ABxxyym2zz*AB.z*rPC.z*shALPHAa[ai]*eta + 2*ABxxyym2zz*AB.z*rPC.z*shALPHAb[bi]*eta + AB.x*rPC.x*(shALPHAa[ai] - shALPHAb[bi])*(-1 + ABxxyym2zz*eta) + AB.y*rPC.y*(shALPHAa[ai] - shALPHAb[bi])*(-1 + ABxxyym2zz*eta)) - 12*PC2gamma2*(3 + 8*AB.z*rPC.z*shALPHAa[ai] + ABxxyym2zz*PCxxyym2zz*shALPHAa[ai]*shALPHAa[ai] - 8*AB.z*rPC.z*shALPHAb[bi] + ABxxyym2zz*PCxxyym2zz*shALPHAb[bi]*shALPHAb[bi] + 2*PCxxyym2zz*gamma + 12*rPC.z*rPC.z*gamma - 4*ABxxyym2zz*PCxxyym2zz*gamma*eta - 8*AB.z*AB.z*PCxxyym2zz*gamma*eta + 4*ABxxyym2zz*rPC.y*rPC.y*gamma*eta + 8*AB.z*AB.z*rPC.y*rPC.y*gamma*eta - 8*ABxxyym2zz*rPC.z*rPC.z*gamma*eta - 32*AB.z*AB.z*rPC.z*rPC.z*gamma*eta + 4*AB.y*AB.y*(PCxxyym2zz - 2*rPC.y*rPC.y + 2*rPC.z*rPC.z)*gamma*eta + 2*AB.y*rPC.y*(shALPHAa[ai] - shALPHAb[bi] + 8*AB.z*rPC.z*gamma*eta) + 2*AB.x*rPC.x*(shALPHAa[ai] - shALPHAb[bi] - 4*AB.y*rPC.y*gamma*eta + 8*AB.z*rPC.z*gamma*eta)));
factor+= (2*PC*sqrtf(gamma)*(105*powf(PCxxyym2zz,2)*powf(gamma,2) + 10*PC2gamma*gamma*(-36*rPC.z*rPC.z - 6*PCxxyym2zz*(1 - 2*AB.z*rPC.z*shALPHAa[ai] + AB.x*rPC.x*(shALPHAa[ai] - shALPHAb[bi]) + AB.y*rPC.y*(shALPHAa[ai] - shALPHAb[bi]) + 2*AB.z*rPC.z*shALPHAb[bi]) + 7*powf(PCxxyym2zz,2)*gamma) - 4*PC2gamma2*(-9 - 24*AB.z*rPC.z*shALPHAa[ai] - 3*ABxxyym2zz*PCxxyym2zz*shALPHAa[ai]*shALPHAa[ai] + 24*AB.z*rPC.z*shALPHAb[bi] - 3*ABxxyym2zz*PCxxyym2zz*shALPHAb[bi]*shALPHAb[bi] + 4*PCxxyym2zz*gamma + 24*rPC.z*rPC.z*gamma - 20*AB.z*PCxxyym2zz*rPC.z*shALPHAa[ai]*gamma + 20*AB.z*PCxxyym2zz*rPC.z*shALPHAb[bi]*gamma - 7*powf(PCxxyym2zz,2)*powf(gamma,2) + 12*ABxxyym2zz*PCxxyym2zz*gamma*eta + 24*AB.z*AB.z*PCxxyym2zz*gamma*eta - 12*ABxxyym2zz*rPC.y*rPC.y*gamma*eta - 24*AB.z*AB.z*rPC.y*rPC.y*gamma*eta + 24*ABxxyym2zz*rPC.z*rPC.z*gamma*eta + 96*AB.z*AB.z*rPC.z*rPC.z*gamma*eta - 12*AB.y*AB.y*(PCxxyym2zz - 2*rPC.y*rPC.y + 2*rPC.z*rPC.z)*gamma*eta + 2*AB.y*rPC.y*(shALPHAb[bi]*(3 - 5*PCxxyym2zz*gamma) + shALPHAa[ai]*(-3 + 5*PCxxyym2zz*gamma) - 24*AB.z*rPC.z*gamma*eta) + 2*AB.x*rPC.x*(shALPHAb[bi]*(3 - 5*PCxxyym2zz*gamma) + shALPHAa[ai]*(-3 + 5*PCxxyym2zz*gamma) + 12*(AB.y*rPC.y - 2*AB.z*rPC.z)*gamma*eta)) + 8*PC2gamma3*(-3 - 2*AB.y*PCxxyym2zz*rPC.y*shALPHAa[ai]*gamma + 4*AB.z*PCxxyym2zz*rPC.z*shALPHAa[ai]*gamma + 2*AB.y*PCxxyym2zz*rPC.y*shALPHAb[bi]*gamma - 4*AB.z*PCxxyym2zz*rPC.z*shALPHAb[bi]*gamma + powf(PCxxyym2zz,2)*powf(gamma,2) + 12*AB.z*AB.z*eta + 4*AB.y*AB.y*PCxxyym2zz*gamma*eta - 8*AB.z*AB.z*PCxxyym2zz*gamma*eta - 8*AB.y*AB.y*rPC.y*rPC.y*gamma*eta + 8*AB.z*AB.z*rPC.y*rPC.y*gamma*eta + 16*AB.y*AB.z*rPC.y*rPC.z*gamma*eta + 8*AB.y*AB.y*rPC.z*rPC.z*gamma*eta - 32*AB.z*AB.z*rPC.z*rPC.z*gamma*eta - 2*AB.x*rPC.x*gamma*(PCxxyym2zz*shALPHAa[ai] - PCxxyym2zz*shALPHAb[bi] + 4*AB.y*rPC.y*eta - 8*AB.z*rPC.z*eta) + ABxxyym2zz*(2*(1 - 2*AB.z*rPC.z*shALPHAa[ai] + AB.x*rPC.x*(shALPHAa[ai] - shALPHAb[bi]) + AB.y*rPC.y*(shALPHAa[ai] - shALPHAb[bi]) + 2*AB.z*rPC.z*shALPHAb[bi] + 2*rPC.y*rPC.y*gamma - 4*rPC.z*rPC.z*gamma)*eta + PCxxyym2zz*(shALPHAa[ai]*shALPHAa[ai] + shALPHAb[bi]*shALPHAb[bi] - 4*gamma*eta)))))*expf(-PC2gamma);

							factor*= 0.315392*0.315392;
							factor *= PI/(16.*PC2gamma2*PC2gamma3*gamma*gamma*sqrtf(gamma));

						}
						else if(shALMOa.z == 0 && shALMOb.z == 1) {

factor = -(ERFgPC*PI_1over2*(16*AB.z*AB.z*PC2gamma2*rPC.z*(2*AB.x*PC2gamma*shALPHAa[ai] - 3*rPC.x*gamma)*eta + rPC.z*(-3*rPC.x*(4*ABxxyym2zz*PC2gamma2*shALPHAb[bi]*shALPHAb[bi] + gamma*(10*PC2gamma - 4*PC2gamma2 + 20*AB.y*PC2gamma*rPC.y*shALPHAb[bi] + 35*PCxxyym2zz*gamma)) + 24*AB.x*AB.x*PC2gamma2*rPC.x*gamma*eta + 2*AB.x*PC2gamma*(15*(PCxxyym2zz*shALPHAa[ai] - 2*rPC.x*rPC.x*shALPHAb[bi])*gamma - 4*PC2gamma2*(2*shALPHAa[ai] + shALPHAb[bi] - ABxxyym2zz*shALPHAb[bi]*eta) + 6*PC2gamma*(2*shALPHAa[ai] + shALPHAb[bi] + 2*AB.y*rPC.y*gamma*eta))) - 2*AB.z*PC2gamma*(-15*rPC.x*(PCxxyym2zz*shALPHAa[ai] + 4*rPC.z*rPC.z*shALPHAb[bi])*gamma + 8*AB.x*PC2gamma3*eta*(1 + ABxxyym2zz*eta) + 6*PC2gamma*(rPC.x*(shALPHAa[ai] + 2*shALPHAb[bi] - 2*AB.y*rPC.y*gamma*eta) + AB.x*(PCxxyym2zz*shALPHAa[ai]*shALPHAa[ai] - 2*PCxxyym2zz*gamma*eta + 2*rPC.y*rPC.y*gamma*eta)) + PC2gamma2*(4*AB.x*(-1 + 2*AB.y*rPC.y*shALPHAa[ai])*eta + rPC.x*(shALPHAa[ai]*(-4 + 8*AB.x*AB.x*eta) - 4*shALPHAb[bi]*(2 + ABxxyym2zz*eta))))));
factor+= (sqrtf(PC2gamma)*gamma*(-210*rPC.x*PCxxyym2zz*rPC.z*gamma - 4*PC2gamma*(-3*AB.x*(2*PC*(2*rPC.z*shALPHAa[ai] - AB.z*PCxxyym2zz*shALPHAa[ai]*shALPHAa[ai] + rPC.z*shALPHAb[bi]) + 5*rPC.z*(PCxxyym2zz*shALPHAa[ai] - 2*PCxxyym2zz*shALPHAb[bi] + 2*rPC.y*rPC.y*shALPHAb[bi] - 4*rPC.z*rPC.z*shALPHAb[bi])) + rPC.x*(-60*AB.z*rPC.z*rPC.z*shALPHAb[bi] + 3*AB.z*(-5*PCxxyym2zz*shALPHAa[ai] + 2*PC*(shALPHAa[ai] + 2*shALPHAb[bi])) + rPC.z*(15 + 30*AB.y*rPC.y*shALPHAb[bi] + 6*ABxxyym2zz*PC*shALPHAb[bi]*shALPHAb[bi] + 35*PCxxyym2zz*gamma))) - 16*PC2gamma3*(-(PCxxyym2zz*(AB.z*rPC.x + AB.x*rPC.z)*shALPHAa[ai]) + 2*rPC.x*rPC.z*(AB.x*rPC.x + AB.y*rPC.y - 2*AB.z*rPC.z)*shALPHAb[bi] + rPC.x*PCxxyym2zz*rPC.z*gamma - 2*(AB.z*rPC.x + AB.x*rPC.z)*(AB.x*rPC.x + AB.y*rPC.y - 2*AB.z*rPC.z)*eta) + 8*PC2gamma2*(-2*rPC.x*rPC.z + 5*PCxxyym2zz*(AB.z*rPC.x + AB.x*rPC.z)*shALPHAa[ai] - 2*AB.x*AB.z*PC*PCxxyym2zz*shALPHAa[ai]*shALPHAa[ai] - 10*rPC.x*rPC.z*(AB.x*rPC.x + AB.y*rPC.y - 2*AB.z*rPC.z)*shALPHAb[bi] - 2*ABxxyym2zz*PC*rPC.x*rPC.z*shALPHAb[bi]*shALPHAb[bi] - 7*rPC.x*PCxxyym2zz*rPC.z*gamma + 2*AB.x*AB.z*PC*eta + 6*(AB.z*rPC.x + AB.x*rPC.z)*(AB.x*rPC.x + AB.y*rPC.y - 2*AB.z*rPC.z)*eta - 4*AB.x*AB.z*PC*(AB.x*rPC.x + AB.y*rPC.y - 2*AB.z*rPC.z)*shALPHAa[ai]*eta + 2*ABxxyym2zz*PC*(AB.z*rPC.x + AB.x*rPC.z)*shALPHAb[bi]*eta)))*expf(-PC2gamma);

							factor*= 0.315392*1.09255;
							factor *= PI/(16*sqrtf(PC2gamma)*PC2gamma2*PC2gamma2*gamma*gamma*gamma);

						}
						else if(shALMOa.z == 0 && shALMOb.z == 2) {

factor = ERFgPC*PI_1over2*(24*PC2gamma2*(AB.x*rPC.x - AB.y*rPC.y)*shALPHAa[ai] - 16*PC2gamma3*(AB.x*rPC.x - AB.y*rPC.y)*shALPHAa[ai] + 12*(ABxxyym2zz - 2*AB.y*AB.y + 2*AB.z*AB.z)*PC2gamma2*PCxxyym2zz*shALPHAa[ai]*shALPHAa[ai] - 24*PC2gamma2*(AB.x*rPC.x - AB.y*rPC.y)*shALPHAb[bi] + 16*PC2gamma3*(AB.x*rPC.x - AB.y*rPC.y)*shALPHAb[bi] + 12*ABxxyym2zz*PC2gamma2*(PCxxyym2zz - 2*rPC.y*rPC.y + 2*rPC.z*rPC.z)*shALPHAb[bi]*shALPHAb[bi] - 60*PC2gamma*(rPC.x*rPC.x - rPC.y*rPC.y)*gamma + 24*PC2gamma2*(rPC.x*rPC.x - rPC.y*rPC.y)*gamma + 60*PC2gamma*PCxxyym2zz*(-(AB.x*rPC.x) + AB.y*rPC.y)*shALPHAa[ai]*gamma + 60*PC2gamma*(rPC.x*rPC.x - rPC.y*rPC.y)*(AB.x*rPC.x + AB.y*rPC.y - 2*AB.z*rPC.z)*shALPHAb[bi]*gamma + 105*PCxxyym2zz*(PCxxyym2zz - 2*rPC.y*rPC.y + 2*rPC.z*rPC.z)*powf(gamma,2) + 16*(AB.x*AB.x - AB.y*AB.y)*PC2gamma3*eta - 32*(AB.x*AB.x - AB.y*AB.y)*PC2gamma2*PC2gamma2*eta + 16*(AB.x*AB.x - AB.y*AB.y)*PC2gamma3*(AB.x*rPC.x + AB.y*rPC.y - 2*AB.z*rPC.z)*shALPHAa[ai]*eta + 16*ABxxyym2zz*PC2gamma3*(-(AB.x*rPC.x) + AB.y*rPC.y)*shALPHAb[bi]*eta - 48*PC2gamma2*(AB.x*rPC.x - AB.y*rPC.y)*(AB.x*rPC.x + AB.y*rPC.y - 2*AB.z*rPC.z)*gamma*eta + 16*ABxxyym2zz*(ABxxyym2zz - 2*AB.y*AB.y + 2*AB.z*AB.z)*PC2gamma2*PC2gamma2*eta*eta);
factor+= (2*sqrtf(PC2gamma)*gamma*(60*PC2gamma*(rPC.x*rPC.x - rPC.y*rPC.y) + 16*PC2gamma2*(rPC.x*rPC.x - rPC.y*rPC.y) - 24*PC*PC2gamma*(AB.x*rPC.x - AB.y*rPC.y)*shALPHAa[ai] + 60*PC2gamma*PCxxyym2zz*(AB.x*rPC.x - AB.y*rPC.y)*shALPHAa[ai] + 40*PC2gamma2*PCxxyym2zz*(AB.x*rPC.x - AB.y*rPC.y)*shALPHAa[ai] + 16*PC2gamma3*PCxxyym2zz*(AB.x*rPC.x - AB.y*rPC.y)*shALPHAa[ai] - 12*(ABxxyym2zz - 2*AB.y*AB.y + 2*AB.z*AB.z)*PC*PC2gamma*PCxxyym2zz*shALPHAa[ai]*shALPHAa[ai] - 8*(ABxxyym2zz - 2*AB.y*AB.y + 2*AB.z*AB.z)*PC*PC2gamma2*PCxxyym2zz*shALPHAa[ai]*shALPHAa[ai] + 24*PC*PC2gamma*(AB.x*rPC.x - AB.y*rPC.y)*shALPHAb[bi] - 60*PC2gamma*(rPC.x*rPC.x - rPC.y*rPC.y)*(AB.x*rPC.x + AB.y*rPC.y - 2*AB.z*rPC.z)*shALPHAb[bi] - 40*PC2gamma2*(rPC.x*rPC.x - rPC.y*rPC.y)*(AB.x*rPC.x + AB.y*rPC.y - 2*AB.z*rPC.z)*shALPHAb[bi] - 16*PC2gamma3*(rPC.x*rPC.x - rPC.y*rPC.y)*(AB.x*rPC.x + AB.y*rPC.y - 2*AB.z*rPC.z)*shALPHAb[bi] - 12*ABxxyym2zz*PC*PC2gamma*(PCxxyym2zz - 2*rPC.y*rPC.y + 2*rPC.z*rPC.z)*shALPHAb[bi]*shALPHAb[bi] - 8*ABxxyym2zz*PC*PC2gamma2*(PCxxyym2zz - 2*rPC.y*rPC.y + 2*rPC.z*rPC.z)*shALPHAb[bi]*shALPHAb[bi] - 105*PCxxyym2zz*(PCxxyym2zz - 2*rPC.y*rPC.y + 2*rPC.z*rPC.z)*gamma - 70*PC2gamma*PCxxyym2zz*(PCxxyym2zz - 2*rPC.y*rPC.y + 2*rPC.z*rPC.z)*gamma - 28*PC2gamma2*PCxxyym2zz*(PCxxyym2zz - 2*rPC.y*rPC.y + 2*rPC.z*rPC.z)*gamma - 8*PC2gamma3*PCxxyym2zz*(PCxxyym2zz - 2*rPC.y*rPC.y + 2*rPC.z*rPC.z)*gamma - 16*(AB.x*AB.x - AB.y*AB.y)*PC*PC2gamma2*eta + 48*PC2gamma2*(AB.x*rPC.x - AB.y*rPC.y)*(AB.x*rPC.x + AB.y*rPC.y - 2*AB.z*rPC.z)*eta + 32*PC2gamma3*(AB.x*rPC.x - AB.y*rPC.y)*(AB.x*rPC.x + AB.y*rPC.y - 2*AB.z*rPC.z)*eta - 16*(AB.x*AB.x - AB.y*AB.y)*PC*PC2gamma2*(AB.x*rPC.x + AB.y*rPC.y - 2*AB.z*rPC.z)*shALPHAa[ai]*eta + 16*ABxxyym2zz*PC*PC2gamma2*(AB.x*rPC.x - AB.y*rPC.y)*shALPHAb[bi]*eta))*expf(-PC2gamma);

							factor*= 0.315392*0.546274;
							factor *= PI/(16*sqrtf(PC2gamma)*PC2gamma2*PC2gamma2*gamma*gamma*gamma);

						}
						else if(shALMOa.z == 1 && shALMOb.z == 1) {

factor = -(ERFgPC*PI_1over2*(-15*PC2gamma*(rPC.z*rPC.z + rPC.x*rPC.x*(1 + 2*AB.z*rPC.z*(shALPHAa[ai] - shALPHAb[bi])) + 2*AB.x*rPC.x*rPC.z*rPC.z*(shALPHAa[ai] - shALPHAb[bi]))*gamma + 105*rPC.x*rPC.x*rPC.z*rPC.z*powf(gamma,2) + 4*PC2gamma2*PC2gamma2*(-1 + 2*AB.x*AB.x*eta)*(-1 + 2*AB.z*AB.z*eta) + 4*PC2gamma3*(-1 + AB.z*rPC.z*(-shALPHAa[ai] + shALPHAb[bi]) + AB.z*AB.z*eta + AB.x*AB.x*(1 + 2*AB.z*rPC.z*(shALPHAa[ai] - shALPHAb[bi]))*eta + AB.x*rPC.x*(shALPHAa[ai] - shALPHAb[bi])*(-1 + 2*AB.z*AB.z*eta)) - 3*PC2gamma2*(-1 + 2*AB.z*rPC.z*(-shALPHAa[ai] + shALPHAb[bi]) - 2*rPC.x*rPC.x*gamma - 2*rPC.z*rPC.z*gamma + 4*AB.z*AB.z*rPC.x*rPC.x*gamma*eta + 4*AB.x*AB.x*rPC.z*rPC.z*gamma*eta - 2*AB.x*rPC.x*(shALPHAa[ai] + 2*AB.z*rPC.z*shALPHAa[ai]*shALPHAa[ai] - shALPHAb[bi] + 2*AB.z*rPC.z*shALPHAb[bi]*shALPHAb[bi] - 4*AB.z*rPC.z*gamma*eta))));
factor+= (2*sqrtf(PC2gamma)*(105*rPC.x*rPC.x*rPC.z*rPC.z*powf(gamma,2) + 5*PC2gamma*gamma*(-3*rPC.z*rPC.z + 6*AB.x*rPC.x*rPC.z*rPC.z*(-shALPHAa[ai] + shALPHAb[bi]) + rPC.x*rPC.x*(-3 + 6*AB.z*rPC.z*(-shALPHAa[ai] + shALPHAb[bi]) + 14*rPC.z*rPC.z*gamma)) + PC2gamma3*(-2 + 8*AB.z*rPC.x*rPC.x*rPC.z*(-shALPHAa[ai] + shALPHAb[bi])*gamma + 8*rPC.x*rPC.x*rPC.z*rPC.z*powf(gamma,2) + AB.z*AB.z*(4 - 8*rPC.x*rPC.x*gamma)*eta + 4*AB.x*AB.x*(1 + 2*AB.z*rPC.z*(shALPHAa[ai] - shALPHAb[bi]) - 2*rPC.z*rPC.z*gamma)*eta + 8*AB.x*rPC.x*(AB.z*rPC.z*powf(shALPHAa[ai] - shALPHAb[bi],2) + rPC.z*rPC.z*(-shALPHAa[ai] + shALPHAb[bi])*gamma + AB.z*AB.z*(shALPHAa[ai] - shALPHAb[bi])*eta)) + PC2gamma2*(3 - 4*rPC.x*rPC.x*gamma - 4*rPC.z*rPC.z*gamma + 28*rPC.x*rPC.x*rPC.z*rPC.z*powf(gamma,2) - 2*AB.z*rPC.z*(shALPHAa[ai] - shALPHAb[bi])*(-3 + 10*rPC.x*rPC.x*gamma) - 12*AB.z*AB.z*rPC.x*rPC.x*gamma*eta - 12*AB.x*AB.x*rPC.z*rPC.z*gamma*eta + 2*AB.x*rPC.x*(6*AB.z*rPC.z*shALPHAa[ai]*shALPHAa[ai] - 3*shALPHAb[bi] + 6*AB.z*rPC.z*shALPHAb[bi]*shALPHAb[bi] + 10*rPC.z*rPC.z*shALPHAb[bi]*gamma + shALPHAa[ai]*(3 - 10*rPC.z*rPC.z*gamma) - 12*AB.z*rPC.z*gamma*eta))))*expf(-PC2gamma);

							factor*= 1.09255*1.09255;
							factor *= PI/(16*sqrtf(PC2gamma)*PC2gamma2*PC2gamma2*gamma*gamma*gamma);

						}
						else if(shALMOa.z == 1 && shALMOb.z == 2) {

factor = -(ERFgPC*PI_1over2*(12*AB.x*PC2gamma2*rPC.z*shALPHAa[ai] - 8*AB.x*PC2gamma3*rPC.z*shALPHAa[ai] + 12*(AB.x*AB.x - AB.y*AB.y)*PC2gamma2*rPC.x*rPC.z*shALPHAa[ai]*shALPHAa[ai] - 12*AB.z*PC2gamma2*rPC.x*shALPHAb[bi] + 8*AB.z*PC2gamma3*rPC.x*shALPHAb[bi] + 12*AB.x*AB.z*PC2gamma2*(rPC.x*rPC.x - rPC.y*rPC.y)*shALPHAb[bi]*shALPHAb[bi] - 30*PC2gamma*rPC.x*rPC.z*gamma + 12*PC2gamma2*rPC.x*rPC.z*gamma + 60*PC2gamma*rPC.x*(-(AB.x*rPC.x) + AB.y*rPC.y)*rPC.z*shALPHAa[ai]*gamma + 30*PC2gamma*(rPC.x*rPC.x - rPC.y*rPC.y)*(AB.z*rPC.x + AB.x*rPC.z)*shALPHAb[bi]*gamma + 105*rPC.x*(rPC.x*rPC.x - rPC.y*rPC.y)*rPC.z*powf(gamma,2) + 8*AB.x*AB.z*PC2gamma3*eta - 16*AB.x*AB.z*PC2gamma2*PC2gamma2*eta + 8*(AB.x*AB.x - AB.y*AB.y)*PC2gamma3*(AB.z*rPC.x + AB.x*rPC.z)*shALPHAa[ai]*eta + 16*AB.x*AB.z*PC2gamma3*(-(AB.x*rPC.x) + AB.y*rPC.y)*shALPHAb[bi]*eta - 24*PC2gamma2*(AB.x*rPC.x - AB.y*rPC.y)*(AB.z*rPC.x + AB.x*rPC.z)*gamma*eta + 16*AB.x*(AB.x*AB.x - AB.y*AB.y)*AB.z*PC2gamma2*PC2gamma2*eta*eta));
factor+= (2*sqrtf(PC2gamma)*gamma*(105*rPC.x*(rPC.x*rPC.x - rPC.y*rPC.y)*rPC.z*gamma + 2*PC2gamma*(-15*rPC.x*rPC.z + 6*AB.x*PC*rPC.z*shALPHAa[ai] + 30*rPC.x*(-(AB.x*rPC.x) + AB.y*rPC.y)*rPC.z*shALPHAa[ai] + 6*(AB.x*AB.x - AB.y*AB.y)*PC*rPC.x*rPC.z*shALPHAa[ai]*shALPHAa[ai] - 6*AB.z*PC*rPC.x*shALPHAb[bi] + 15*(rPC.x*rPC.x - rPC.y*rPC.y)*(AB.z*rPC.x + AB.x*rPC.z)*shALPHAb[bi] + 6*AB.x*AB.z*PC*(rPC.x*rPC.x - rPC.y*rPC.y)*shALPHAb[bi]*shALPHAb[bi] + 35*rPC.x*(rPC.x*rPC.x - rPC.y*rPC.y)*rPC.z*gamma) - 8*PC2gamma3*(2*rPC.x*(AB.x*rPC.x - AB.y*rPC.y)*rPC.z*shALPHAa[ai] - (rPC.x*rPC.x - rPC.y*rPC.y)*(AB.z*rPC.x + AB.x*rPC.z)*shALPHAb[bi] + rPC.x*(-rPC.x*rPC.x + rPC.y*rPC.y)*rPC.z*gamma + 2*(AB.x*rPC.x - AB.y*rPC.y)*(AB.z*rPC.x + AB.x*rPC.z)*eta) + 4*PC2gamma2*(-2*rPC.x*rPC.z + 10*rPC.x*(-(AB.x*rPC.x) + AB.y*rPC.y)*rPC.z*shALPHAa[ai] + 2*(AB.x*AB.x - AB.y*AB.y)*PC*rPC.x*rPC.z*shALPHAa[ai]*shALPHAa[ai] + 5*(rPC.x*rPC.x - rPC.y*rPC.y)*(AB.z*rPC.x + AB.x*rPC.z)*shALPHAb[bi] + 2*AB.x*AB.z*PC*(rPC.x*rPC.x - rPC.y*rPC.y)*shALPHAb[bi]*shALPHAb[bi] + 7*rPC.x*(rPC.x*rPC.x - rPC.y*rPC.y)*rPC.z*gamma + 2*AB.x*AB.z*PC*eta - 6*(AB.x*rPC.x - AB.y*rPC.y)*(AB.z*rPC.x + AB.x*rPC.z)*eta + 2*(AB.x*AB.x - AB.y*AB.y)*PC*(AB.z*rPC.x + AB.x*rPC.z)*shALPHAa[ai]*eta + 4*AB.x*AB.z*PC*(-(AB.x*rPC.x) + AB.y*rPC.y)*shALPHAb[bi]*eta)))*expf(-PC2gamma);

							factor*= 1.09255*0.546274;
							factor *= PI/(16*sqrtf(PC2gamma)*PC2gamma2*PC2gamma2*gamma*gamma*gamma);

						}
						else if(shALMOa.z == 2 && shALMOb.z == 2) {

factor = -(ERFgPC*PI_1over2*(-60*PC2gamma*(rPC.x*rPC.x + rPC.y*rPC.y + (AB.x*rPC.x - AB.y*rPC.y)*(rPC.x*rPC.x - rPC.y*rPC.y)*shALPHAa[ai] - (AB.x*rPC.x - AB.y*rPC.y)*(rPC.x*rPC.x - rPC.y*rPC.y)*shALPHAb[bi])*gamma + 105*powf(rPC.x*rPC.x - rPC.y*rPC.y,2)*powf(gamma,2) + 12*PC2gamma2*(1 + 2*(AB.x*rPC.x + AB.y*rPC.y)*shALPHAa[ai] + (AB.x*AB.x - AB.y*AB.y)*(rPC.x*rPC.x - rPC.y*rPC.y)*shALPHAa[ai]*shALPHAa[ai] - 2*(AB.x*rPC.x + AB.y*rPC.y)*shALPHAb[bi] + (AB.x*AB.x - AB.y*AB.y)*(rPC.x*rPC.x - rPC.y*rPC.y)*shALPHAb[bi]*shALPHAb[bi] + 2*(PCxxyym2zz + 2*rPC.z*rPC.z)*gamma - 4*powf(AB.x*rPC.x - AB.y*rPC.y,2)*gamma*eta) + 16*PC2gamma2*PC2gamma2*(1 - 2*(ABxxyym2zz + 2*AB.z*AB.z)*eta + powf(AB.x*AB.x - AB.y*AB.y,2)*eta*eta) + 16*PC2gamma3*(powf(AB.x,3)*rPC.x*(shALPHAa[ai] - shALPHAb[bi])*eta + AB.x*AB.x*(1 + AB.y*rPC.y*(-shALPHAa[ai] + shALPHAb[bi]))*eta + (1 + AB.y*rPC.y*(shALPHAa[ai] - shALPHAb[bi]))*(-1 + AB.y*AB.y*eta) - AB.x*rPC.x*(shALPHAa[ai] - shALPHAb[bi])*(1 + AB.y*AB.y*eta))));
factor+= (2*sqrtf(PC2gamma)*(105*powf(rPC.x*rPC.x - rPC.y*rPC.y,2)*powf(gamma,2) + 10*PC2gamma*gamma*(-6*rPC.x*rPC.x - 6*rPC.y*rPC.y - 6*(AB.x*rPC.x - AB.y*rPC.y)*(rPC.x*rPC.x - rPC.y*rPC.y)*shALPHAa[ai] + 6*(AB.x*rPC.x - AB.y*rPC.y)*(rPC.x*rPC.x - rPC.y*rPC.y)*shALPHAb[bi] + 7*powf(rPC.x*rPC.x - rPC.y*rPC.y,2)*gamma) + 4*PC2gamma2*(3 + 6*(AB.x*rPC.x + AB.y*rPC.y)*shALPHAa[ai] + 3*(AB.x*AB.x - AB.y*AB.y)*(rPC.x*rPC.x - rPC.y*rPC.y)*shALPHAa[ai]*shALPHAa[ai] - 6*(AB.x*rPC.x + AB.y*rPC.y)*shALPHAb[bi] + 3*(AB.x*AB.x - AB.y*AB.y)*(rPC.x*rPC.x - rPC.y*rPC.y)*shALPHAb[bi]*shALPHAb[bi] - 4*(PCxxyym2zz + 2*rPC.z*rPC.z)*gamma - 10*(AB.x*rPC.x - AB.y*rPC.y)*(rPC.x*rPC.x - rPC.y*rPC.y)*shALPHAa[ai]*gamma + 10*(AB.x*rPC.x - AB.y*rPC.y)*(rPC.x*rPC.x - rPC.y*rPC.y)*shALPHAb[bi]*gamma + 7*powf(rPC.x*rPC.x - rPC.y*rPC.y,2)*powf(gamma,2) - 12*powf(AB.x*rPC.x - AB.y*rPC.y,2)*gamma*eta) + 8*PC2gamma3*(-1 + (AB.x*AB.x - AB.y*AB.y)*(rPC.x*rPC.x - rPC.y*rPC.y)*shALPHAa[ai]*shALPHAa[ai] + (AB.x*AB.x - AB.y*AB.y)*(rPC.x*rPC.x - rPC.y*rPC.y)*shALPHAb[bi]*shALPHAb[bi] - 2*(AB.x*rPC.x - AB.y*rPC.y)*(rPC.x*rPC.x - rPC.y*rPC.y)*shALPHAa[ai]*gamma + 2*(AB.x*rPC.x - AB.y*rPC.y)*(rPC.x*rPC.x - rPC.y*rPC.y)*shALPHAb[bi]*gamma + powf(rPC.x*rPC.x - rPC.y*rPC.y,2)*powf(gamma,2) + 2*(ABxxyym2zz + 2*AB.z*AB.z)*eta + 2*(AB.x*AB.x - AB.y*AB.y)*(AB.x*rPC.x - AB.y*rPC.y)*shALPHAa[ai]*eta - 2*(AB.x*AB.x - AB.y*AB.y)*(AB.x*rPC.x - AB.y*rPC.y)*shALPHAb[bi]*eta - 4*powf(AB.x*rPC.x - AB.y*rPC.y,2)*gamma*eta)))*expf(-PC2gamma);

							factor*= 0.546274*0.546274;
							factor *= PI/(16*sqrtf(PC2gamma)*PC2gamma2*PC2gamma2*gamma*gamma*gamma);
						}
					}
					

					factor *= expAB;
					factor *= shCOEFFa[ai]*shCOEFFb[bi];
					product += factor;
					__syncthreads();
				}
				__syncthreads();
			} // end of loop over GTO contractors
			__syncthreads();

			product *= 2; // this is cos the DM only has spin up
			if(p != q) product *= 2; // this is because we loop over the upper triangle of DM only
			product *= shDM[q];

			//if(sidx == 0)

			V += product;
		}
	}

	// compute the write index
	sidx = (threadIdx.x + blockIdx.x*B);
	sidx+= (threadIdx.y + blockIdx.y*B) * gridDim.x * B;
	sidx+= (threadIdx.z + blockIdx.z*B) * gridDim.x * gridDim.y * B_2;

	float r, c;
	for(ushort i=0; i<NATM; i++) {

		c = rC.x - scoords[i].x; r = c*c;
		c = rC.y - scoords[i].y; r+= c*c;
		c = rC.z - scoords[i].z; r+= c*c;
		
		r = sqrtf(r);
		if(r > 1.0E-8)
			V += stypes[i] / r;

		// this assumes the nuclear charge is in chi_a^2, an uncontracted s GTO with a=10000
		//r = sqrtf(r*10000);
		//V -= 2.4543692606170252E-14 * stypes[i] * erff(r)/r;
	}
	

	Vout[sidx] = V;
}


