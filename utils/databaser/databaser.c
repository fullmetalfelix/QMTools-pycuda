#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>


#define ANG2BOR 1.8897259886
#define MAXAOS 15
#define MAXAOC 20
#define M_PI 3.14159265358979323846
#define OUTSIZE 6


#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))


typedef struct float3 float3;
struct float3 {
	float x, y, z;
};

typedef struct molecule molecule;
struct molecule {

	int 	natm;
	int 	*Zs;
	float 	*xyz;

	int 	norb;
	float 	*dm;
	short 	*almos;


	float *alphas, *coeffs;


};


int nshells;
int shellreps;



float3 vector_add(float3* a, float3* b){
	float3 r;
	r.x = a->x + b->x;
	r.y = a->y + b->y;
	r.z = a->z + b->z;

	return r;
}


float SolidHarmonicR(short L, short m, float3 r) {

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

float IntegralFunction(float rac2, float dx, int Zc) {

	float acc = 0;
	float alpha = rac2/(dx*dx);
	float f0 = 1;
	float f1, dt=0.001f;
	float t1;


	for(int i=1;i<=1000;i++){

		t1 = i*dt;
		f1 = exp(-alpha*t1*t1);

		acc += (f1+f0)*dt*0.5f;
		f0 = f1;
	}

	acc = -acc * Zc * M_PI * dx * dx;
	return acc;
}


// TODO: make this for cpu instead
/*
	take some points along the connecting line between atoms, and offline
	take some point in a spherical volume around the atoms
	take some points far away from everything


*/


//	float dxcharge = 0, dycharge = 0;
	//float ddxcharge = 0, ddycharge = 0;


// atomic coords are in BOHR


// compute the density in one point in space
float compute_density(molecule *mol, float3 v, float dx, int nsubdivs) {

	float charge = 0;
	float3 v0;
	float3 vp;

	// lateral size of the subgrid voxel
	float ddx = dx / nsubdivs;

	// inverse volume of the subgrid voxel
	float SUBGRIDiV = 1.0f/(ddx*ddx*ddx);
	float SUBGRIDV = ddx*ddx*ddx;


	float *ap, *cp;
	float *aq, *cq;

	int norb = mol->norb;
	short *almos = mol->almos;




	for(uint16_t p=0; p<norb; ++p) { // loop over DM rows

		ap = &(mol->alphas[almos[p*4 + 3]]);
		cp = &(mol->coeffs[almos[p*4 + 3]]);

		for(uint16_t q=0; q<=p; ++q) { // loop over DM columns

			float3 voxpos;

			aq = &(mol->alphas[mol->almos[q*4 + 3]]);
			cq = &(mol->coeffs[mol->almos[q*4 + 3]]);

			for(uint16_t ix=0; ix<nsubdivs; ix++) {

				vp.x = v.x - 0.5*dx + (ix+0.5)*ddx;

				for(uint16_t iy=0; iy<nsubdivs; iy++) {

					vp.y = v.y - 0.5*dx + (iy+0.5)*ddx;

					for(uint16_t iz=0; iz<nsubdivs; iz++) {

						vp.z = v.z - 0.5*dx + (iz+0.5)*ddx;

						// dm[p,q] * sump (coeff[i] exp[-alpha[i] r**2] Ylm) * sumq (coeff[i] exp[-alpha[i] r**2] Ylm)

						float partial = mol->dm[p*norb + q];

						float3 r; // = scoords[shALMOs[p].x];
						r.x = vp.x - mol->xyz[almos[p*4]*3 + 0];
						r.y = vp.y - mol->xyz[almos[p*4]*3 + 1];
						r.z = vp.z - mol->xyz[almos[p*4]*3 + 2];
						
						partial *= SolidHarmonicR(almos[p*4+1], almos[p*4+2], r);

						// multiply by the contracted gaussians
						float d = r.x*r.x + r.y*r.y + r.z*r.z;
						float acc = 0;
						for(uint16_t ai=0; ai<MAXAOC; ai++) {
							//r.z = alphas[shALMOs[p].w+ai];
							acc += cp[ai] * exp(-ap[ai] * d);
						}
						partial *= acc;

						//r = scoords[shALMOs[q].x];
						r.x = vp.x - mol->xyz[almos[q*4]*3 + 0];
						r.y = vp.y - mol->xyz[almos[q*4]*3 + 1];
						r.z = vp.z - mol->xyz[almos[q*4]*3 + 2];
						partial *= SolidHarmonicR(almos[q*4+1], almos[q*4+2], r);

						d = r.x*r.x + r.y*r.y + r.z*r.z;
						acc = 0;
						for(uint16_t ai=0; ai<MAXAOC; ai++) {
							//r.z = alphas[shALMOs[q].w+ai];
							acc += cq[ai] * exp(-aq[ai] * d);
						}
						partial *= acc;

						if(p != q) partial*=2;

						//printf("%i %i %f, %f\n",p,q,mol->dm[p*norb + q],partial);
						//charge += partial * SUBGRIDiV;
						charge += partial * SUBGRIDV;
					}
				}
			}// end of subgrid loop
		}
	}

	// this accounts for closed shell (2 electrons per orbital)
	// and multiplied by the voxel volume (integral of the wfn)
	charge = 2*charge; //*dx*dx*dx;
	return charge;
}

// computes nuclear potential in one spot felt by a gaussian charge density.
float compute_VNe(molecule *mol, float3 v, float3 d, float dx) {

	float tx,ty,tz;
	float acc = 0;

	int natm = mol->natm;
	for(uint16_t i=0; i<natm; i++) {
		
		tx = v.x+d.x-mol->xyz[3*i+0];
		ty = v.y+d.y-mol->xyz[3*i+1];
		tz = v.z+d.z-mol->xyz[3*i+2];

		tx = tx*tx + ty*ty + tz*tz;

		acc += IntegralFunction(tx, dx, mol->Zs[i]);
	}

	return acc;
}



void compute_VNe_grads(molecule *mol, float3 p, float dx, float *output) {

	float3 d; d.x = 0; d.y = 0; d.z = 0;
	float VNe[7];
	float step = dx * 0.1f;


	VNe[0] = compute_VNe(mol, p, d, dx);
	
	d.x = step;
	VNe[1] = compute_VNe(mol, p, d, dx);
	d.x = -step;
	VNe[2] = compute_VNe(mol, p, d, dx);
	d.x = 0;

	d.y = step;
	VNe[3] = compute_VNe(mol, p, d, dx);
	d.y = -step;
	VNe[4] = compute_VNe(mol, p, d, dx);
	d.y = 0;

	d.z = step;
	VNe[5] = compute_VNe(mol, p, d, dx);
	d.z = -step;
	VNe[6] = compute_VNe(mol, p, d, dx);
	d.z = 0;


	output[0] = VNe[0];

	float gx = (VNe[1]-VNe[2]);
	float gy = (VNe[3]-VNe[4]);
	float gz = (VNe[5]-VNe[6]);

	output[1] = sqrt(gx*gx + gy*gy + gz*gz)/(2*step);

	gx = VNe[1] - 2*VNe[0] + VNe[2];
	gy = VNe[3] - 2*VNe[0] + VNe[4];
	gz = VNe[5] - 2*VNe[0] + VNe[6];

	output[2] = (gx + gy + gz) / (step*step);
}


int compute_output(molecule *mol, float dx, int nsubdivs, uint32_t seed, float *output) {


	srand(seed);

	float3 p, q;
	float3 pp;

	int c = 0;

	float3 d; 
	float *o = output;

	float rmax = 0;


	// take point on top of the atoms and around them
	for(int i=0; i<mol->natm; i++) {
	  //printf("atom %i\n",i);
		p.x = mol->xyz[3*i+0];
		p.y = mol->xyz[3*i+1];
		p.z = mol->xyz[3*i+2];

		rmax = MAX(rmax, abs(p.x));
		rmax = MAX(rmax, abs(p.y));
		rmax = MAX(rmax, abs(p.z));

		// compute on atoms
		compute_VNe_grads(mol, p, dx, o); o += 3;
		o[0] = compute_density(mol, p, dx, nsubdivs); o += 1;
		c++;

		// compute around them
		for(int s=0; s<nshells; s++) {

			float r0 = (s) * 1.0f;
			float r1 = (s+1) * 1.0f;

			for(int rep=0; rep<shellreps; rep++) {

				float theta = ((float)rand() / (float)RAND_MAX) * M_PI;
				float phi = ((float)rand() / (float)RAND_MAX) * 2 * M_PI;
				float r = ((float)rand() / (float)RAND_MAX) * (r1-r0) + r0;

				
				pp.x = r * sin(theta) * cos(phi);
				pp.y = r * sin(theta) * sin(phi);
				pp.z = r * cos(theta);

				pp = vector_add(&p, &pp);

				compute_VNe_grads(mol, pp, dx, o); o += 3;
				o[0] = compute_density(mol, pp, dx, nsubdivs); o += 1;
				c++;
			}
		}
	}

	rmax += 6;

	{
		// compute a very far away shell
		float r0 = rmax;
		float r1 = 2 * rmax;

		for(int rep=0; rep<shellreps; rep++) {

			float theta = ((float)rand() / (float)RAND_MAX) * M_PI;
			float phi = ((float)rand() / (float)RAND_MAX) * 2 * M_PI;
			float r = ((float)rand() / (float)RAND_MAX) * (r1-r0) + r0;

			
			pp.x = r * sin(theta) * cos(phi);
			pp.y = r * sin(theta) * sin(phi);
			pp.z = r * cos(theta);

			pp = vector_add(&p, &pp);

			compute_VNe_grads(mol, pp, dx, o); o += 3;
			o[0] = compute_density(mol, pp, dx, nsubdivs); o += 1;
			c++;
		}
	}


	for(int i=0; i<mol->natm; i++) {

		p.x = mol->xyz[3*i+0];
		p.y = mol->xyz[3*i+1];
		p.z = mol->xyz[3*i+2];

		// along bonds
		for(int j=0; j<i; j++) {

			q.x = mol->xyz[3*j+0];
			q.y = mol->xyz[3*j+1];
			q.z = mol->xyz[3*j+2];

			float dist = (p.x-q.x)*(p.x-q.x) + (p.y-q.y)*(p.y-q.y) + (p.z-q.z)*(p.z-q.z);
			if(dist >= 36) continue;
			dist = sqrt(dist);

			for(int rep=0; rep<shellreps; rep++) {

				float t = ((float)rand() / (float)RAND_MAX);

				pp.x = p.x*(t-1) + q.x*t + ((float)rand() / (float)RAND_MAX) * 0.25;
				pp.y = p.y*(t-1) + q.y*t + ((float)rand() / (float)RAND_MAX) * 0.25;
				pp.z = p.z*(t-1) + q.z*t + ((float)rand() / (float)RAND_MAX) * 0.25;

				compute_VNe_grads(mol, pp, dx, o); o += 3;
				o[0] = compute_density(mol, pp, dx, nsubdivs); o += 1;
				c++;
			}
		}
	}

	return c;
}



/*
void gpu_densityqube_shmem_subgrid(
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
				voxpos.x = (blockIdx.x * B + threadIdx.x)*dx + (ix*SUBGRIDDX + SUBGRIDDX2)*dx;
				for(ushort iy=0; iy<SUBGRID; iy++) {
					voxpos.y = (blockIdx.y * B + threadIdx.y)*dx + (iy*SUBGRIDDX + SUBGRIDDX2)*dx;
					for(ushort iz=0; iz<SUBGRID; iz++) {
						voxpos.z = (blockIdx.z * B + threadIdx.z)*dx + (iz*SUBGRIDDX + SUBGRIDDX2)*dx;

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

*/
