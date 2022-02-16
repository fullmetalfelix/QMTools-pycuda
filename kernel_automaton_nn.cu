#include "kernel_common.h"

#define NPTS PYCUDA_NPTS
#define NX PYCUDA_NX
#define NY PYCUDA_NY
#define NZ PYCUDA_NZ
//#define ADS PYCUDA_ADS
//#define DIF PYCUDA_DIF
#define nf 4


#define OneOverSqrt2 0.7071067811865475
#define OneOverSqrt3 0.5773502691896258
#define OneOverDIFFTOT 0.05234482976098482

#define DIFFQ 0.02
#define DIFFA 0.02
#define DIFFB 0.02

#define DNASIZE PYCUDA_DNASIZE

#define NL1 PYCUDA_NL1
#define NL2 PYCUDA_NL2
#define ABS_A PYCUDA_ABS_A
#define ABS_B PYCUDA_ABS_B


__constant__ float cParams[DNASIZE];



__global__ void __launch_bounds__(512, 4) gpu_automaton_nn_evolve(
	float 	*qube, 	// input multifield qube
	float 	*qout 	// output multifield qube
	) {


	// can store up to 16 floats per thread in shmem if we use an A100!!!

	volatile __shared__ float buffer1[16*B_3];
	volatile float buffer2[16];

	float final[3];
	final[0] = 0; final[1] = 0; final[2] = 0;
	volatile uint ridx, widx;
	volatile float q1;

	/*ridx = (threadIdx.x + blockIdx.x*B) + (threadIdx.y + blockIdx.y*B)*NX + (threadIdx.z + blockIdx.z)*NX*NY;
	if(ridx == 0){
		for(ushort i=0; i<DNASIZE; i++){
			printf("cmem[%i\t%i] = %f\n",i,ridx,cParams[i]);
		}
	}*/

	// this is a loop over neigbours
	for(ushort deltas=1; deltas < 64; deltas++) {

		if((deltas & 3) == 2) deltas++;
		if((deltas & 12) == 8) deltas+=4;
		if((deltas & 48) == 32) deltas+=16;

		volatile float acc;
		volatile float q2;
		volatile short3 r;

		// to make sure itz antisymm, we run the code twice


		volatile float tQ=0, tA=0, tB=0;
		for(short rep=1; rep>=-1; rep-=2) { // repeat twice - with different sign

			// load the data of the thread assigned voxel - 4 inputs
			{
				r.x = threadIdx.x + blockIdx.x*B;
				r.y = threadIdx.y + blockIdx.y*B;
				r.z = threadIdx.z + blockIdx.z*B;
				ridx = r.x + r.y*NX + r.z*NX*NY;
				widx = (threadIdx.x + threadIdx.y*B + threadIdx.z*B_2)*16;
				#pragma unroll
				for(ushort k=0; k<nf; k++)
					buffer1[widx + k] = qube[ridx + k*NPTS];
				q1 = buffer1[widx];
			}

			// load the neigbour point - other 4 inputs
			{
				// dx is in the first 2 bits of deltas
				short dx;

				dx = (deltas & 1)>0;
				if((deltas & 2)>0) dx *= -1;
				r.x += dx;
				if(r.x == -1) r.x = NX-1;
				if(r.x == NX) r.x = 0;

				dx = (deltas & 4)>0;
				if((deltas & 8)>0) dx *= -1;
				r.y += dx;
				if(r.y == -1) r.y = NY-1;
				if(r.y == NY) r.y = 0;

				dx = (deltas & 16)>0;
				if((deltas & 32)>0) dx *= -1;		
				r.z += dx;
				if(r.z == -1) r.z = NZ-1;
				if(r.z == NZ) r.z = 0;

				ridx = r.x + r.y*gridDim.x*B + r.z*gridDim.x*gridDim.y*B_2;
				widx += 4;
				#pragma unroll
				for(ushort k=0; k<nf; k++)
					buffer1[widx + k] = qube[ridx + k*NPTS];
				q2 = buffer1[widx];
				widx -= 4; // back to the beginning of the input array for this thread buffer1[widx] = first input
			}

			// put some extra constants in the last 8 inputs - buffer1[widx + 8 + k] k=0-7(incl)
			// PYCUDA_GENETICPROGRAM_CONSTS_1
			for(ushort k=0; k<8; k++) buffer1[widx + 8 + k] = cParams[k];
			uint offset = 8;

			// if we are doing the run with flipped inputs, flip them
			if(rep < 0){
				for(ushort k=0; k<nf; k++){
					acc = buffer1[widx + k + 4];
					buffer1[widx + k + 4] = buffer1[widx + k];
					buffer1[widx + k] = acc;
				}
				//acc = q1; q1 = q2; q2 = acc;
			}
			__syncthreads();

			// main NN part
			for(ushort nl=0; nl<NL1; nl++) {
				for(ushort k=0; k<16; k++) { // loop over neurons

					acc = 0;
					for(ushort l=0; l<16; l++)
						acc += cParams[offset+16*k+l] * buffer1[widx + l];
					acc += cParams[offset+16*k+16];
					buffer2[k] = tanhf(acc);

					offset += 16+1;
				}
				// copy outputs to input buffer
				for(ushort k=0; k<16; k++)
					buffer1[widx+k] = buffer2[k];
			}

			// last layer - linear
			for(ushort k=0; k<3; k++) { // loop over neurons

				acc = 0;
				for(ushort l=0; l<16; l++)
					acc += cParams[offset+16*k+l] * buffer1[widx + l];
				acc += cParams[offset+16*k+16];
				buffer2[k] = acc;

				offset += 16+1;
			}

			// now buffer2[0:3] has the transfer factors for the q A B
			// the transfer is T(vox1,vox2) or T(vox2,vox1)

			tQ += rep * buffer2[0];
			tA += rep * buffer2[1];
			tB += rep * buffer2[2];

			/* DEBUG PRINT
			if(q1 != 0 && q2 !=0){
				r.x = threadIdx.x + blockIdx.x*B;
				r.y = threadIdx.y + blockIdx.y*B;
				r.z = threadIdx.z + blockIdx.z*B;
				ridx = r.x + r.y*NX + r.z*NX*NY;
				if(ridx == 450787)
					printf("%i %f %f - delta%i - rep%i buf %f tq %f \n",ridx,q1,q2,deltas,rep, buffer2[0], tQ);
			}
			*/
		}

		buffer2[0] = tanhf(tQ);
		buffer2[1] = tanhf(tA);
		buffer2[2] = tanhf(tB);
		if(buffer2[0] < 0) buffer2[0] *= q1;
		else buffer2[0] *= q2;

		// scale by distance
		float factor = (deltas & 1) + ((deltas & 4)>0) + ((deltas & 16)>0);
		factor = rsqrtf(factor);

		final[0] += buffer2[0] * factor * DIFFQ;
		final[1] += buffer2[1] * factor * DIFFA;
		final[2] += buffer2[2] * factor * DIFFB;
	}

	// compute write outputs
	final[0] = q1 + final[0] * OneOverDIFFTOT; // charge
	final[1] =      final[1] * OneOverDIFFTOT; // A
	final[2] =      final[2] * OneOverDIFFTOT; // B
	// since transfers of A/B are not normalised, there can be field created/destroyed
	// NOOOOOooo....! A/B are initialised to 0, and this GP is antisymmetric => there will never be transfer of nothing!
	// we need a small GP to create/destroy the field based on local info


	// load local info
	volatile short3 r;
	volatile float acc;
	r.x = threadIdx.x + blockIdx.x*B;
	r.y = threadIdx.y + blockIdx.y*B;
	r.z = threadIdx.z + blockIdx.z*B;
	ridx = r.x + r.y*NX + r.z*NX*NY;
	widx = (threadIdx.x + threadIdx.y*B + threadIdx.z*B_2)*16;
	#pragma unroll
	for(ushort k=0; k<nf; k++) buffer1[widx + k] = qube[ridx + k*NPTS];

	qout[ridx +   NPTS] = buffer1[widx + 1]; // copy the VNe as it is
	float fieldA = buffer1[widx + 2];
	float fieldB = buffer1[widx + 3];

	uint offset = 16*16*4+16*4+16*3+3;

	// add 12 constants
	for(ushort k=0; k<12; k++) buffer1[widx + 4 + k] = cParams[offset+k];
	offset += 12;


	// main NN part
	for(ushort nl=0; nl<NL2; nl++) {
		for(ushort k=0; k<16; k++) { // loop over neurons

			acc = 0;
			for(ushort l=0; l<16; l++)
				acc += cParams[offset+16*k+l] * buffer1[widx + l];
			acc += cParams[offset+16*k+16];
			buffer2[k] = tanhf(acc);

			offset += 16+1;
		}
		// copy outputs to input buffer
		for(ushort k=0; k<16; k++)
			buffer1[widx+k] = buffer2[k];
	}

	// last layer - linear
	for(ushort k=0; k<2; k++) { // loop over neurons

		acc = 0;
		for(ushort l=0; l<16; l++)
			acc += cParams[offset+16*k+l] * buffer1[widx + l];
		acc += cParams[offset+16*k+16];
		buffer2[k] = acc;

		offset += 16+1;
	}
	

	// now we have outputs in buffer2[0/1]
	final[1] += buffer2[0] - ABS_A*fieldA;
	final[2] += buffer2[1] - ABS_B*fieldB;

	// save the final results
	// only Q, A and B channels are updated
	qout[ridx] = final[0];
	qout[ridx + 2*NPTS] = final[1];
	qout[ridx + 3*NPTS] = final[2];
}