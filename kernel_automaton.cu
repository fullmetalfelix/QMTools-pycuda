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


__constant__ float cParams[DNASIZE];



__global__ void __launch_bounds__(512, 4) gpu_automaton_evolve(
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

			/* DEBUG PRINT
			if(q1 != 0 && q2 !=0){
				r.x = threadIdx.x + blockIdx.x*B;
				r.y = threadIdx.y + blockIdx.y*B;
				r.z = threadIdx.z + blockIdx.z*B;
				ridx = r.x + r.y*NX + r.z*NX*NY;
				if(ridx == 450787){
					printf("INPUT %i %f %f - rep%i --- %f %f %f %f %f %f %f %f\n",ridx,q1,q2,rep,
						buffer1[widx+0],buffer1[widx+1],buffer1[widx+2],buffer1[widx+3],
						buffer1[widx+4],buffer1[widx+5],buffer1[widx+6],buffer1[widx+7]);
				}
			}*/

			// execute the whole program
			// PYCUDA_GENETICPROGRAM_1


			// now buffer2[] has the transfer factors for the q V A B
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
	// NOOOOOooo....! A/B are initialised to 0, and this GP is antisymmetric => there will never be transfer of anything!
	// we need a small GP to create/destroy the field based on local info


	// load local info - q V A B
	volatile short3 r;
	volatile float acc;
	r.x = threadIdx.x + blockIdx.x*B;
	r.y = threadIdx.y + blockIdx.y*B;
	r.z = threadIdx.z + blockIdx.z*B;
	ridx = r.x + r.y*NX + r.z*NX*NY;
	widx = (threadIdx.x + threadIdx.y*B + threadIdx.z*B_2)*16;
	#pragma unroll
	for(ushort k=0; k<nf; k++) buffer1[widx + k] = qube[ridx + k*NPTS];

	// add 12 constants
	// PYCUDA_GENETICPROGRAM_CONSTS_2

	
	// PYCUDA_GENETICPROGRAM_2



	// now we have outputs in buffer2[0/1]
	final[1] += buffer2[0];
	final[2] += buffer2[1];

	// save the final results
	// only Q, A and B channels are updated
	qout[ridx] = final[0];
	qout[ridx + 1*NPTS] = buffer1[widx + 1]; // copy the VNe as it is
	qout[ridx + 2*NPTS] += final[1];
	qout[ridx + 3*NPTS] += final[2];
}

// this one uses only a 2 component grid for density and VNe
__global__ void __launch_bounds__(512, 4) gpu_automaton_simple_evolve(
	float 	*qube, 	// input multifield qube
	float 	*qout 	// output multifield qube
	) {


	// can store up to 16 floats per thread in shmem if we use an A100!!!

	volatile __shared__ float buffer1[16*B_3];
	volatile float buffer2[16];

	float final = 0;
	volatile uint ridx, widx;
	volatile float q1, v1;

	// load q and V in this point
	widx = (threadIdx.x + threadIdx.y*B + threadIdx.z*B_2)*16;
	ridx = (threadIdx.x + blockIdx.x*B) + (threadIdx.y + blockIdx.y*B)*NX + (threadIdx.z + blockIdx.z*B)*NX*NY;
	q1 = qube[ridx + 0*NPTS];
	v1 = qube[ridx + 1*NPTS];

	// this is a loop over neigbours
	for(ushort deltas=1; deltas < 64; deltas++) {

		if((deltas & 3) == 2) deltas++;
		if((deltas & 12) == 8) deltas+=4;
		if((deltas & 48) == 32) deltas+=16;

		volatile float acc;
		volatile float q2, v2;
		volatile short3 r;

		// load the neigbour point - other 2 inputs
		{
			// dx is in the first 2 bits of deltas
			short dx;

			r.x = threadIdx.x + blockIdx.x*B;
			r.y = threadIdx.y + blockIdx.y*B;
			r.z = threadIdx.z + blockIdx.z*B;

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
			
			q2 = qube[ridx + 0*NPTS];
			v2 = qube[ridx + 1*NPTS];
		}



		// to make sure itz antisymm, we run the code twice
		volatile float tQ=0;
		for(short rep=1; rep>=-1; rep-=2) { // repeat twice - with different sign

			// put the 2/2 inputs in the buffer
			// if we are doing the run with flipped inputs, flip them
			if(rep > 0){
				// correct order
				buffer1[widx + 0] = q1;
				buffer1[widx + 1] = v1;
				buffer1[widx + 2] = q2;
				buffer1[widx + 3] = v2;
			} else {
				// flipped order
				buffer1[widx + 0] = q2;
				buffer1[widx + 1] = v2;
				buffer1[widx + 2] = q1;
				buffer1[widx + 3] = v1;
			}

			// put some extra constants in the last 12 inputs - buffer1[widx + 4 + k] k=0-7(incl)
			// PYCUDA_GENETICPROGRAM_CONSTS_1


			__syncthreads();


			// execute the whole program
			// PYCUDA_GENETICPROGRAM_1


			// now buffer2[] has the transfer factors for the q V A B
			// the transfer is T(vox1,vox2) or T(vox2,vox1)

			tQ += rep * buffer2[0];
		}

		tQ = tanhf(tQ);
		if(tQ < 0) tQ *= q1;
		else tQ *= q2;

		// scale by distance
		float factor = (deltas & 1) + ((deltas & 4)>0) + ((deltas & 16)>0);
		factor = rsqrtf(factor);

		final += tQ * factor * DIFFQ;
	}

	final = q1 + final * OneOverDIFFTOT;
	
	// save the final results
	ridx = (threadIdx.x + blockIdx.x*B) + (threadIdx.y + blockIdx.y*B)*NX + (threadIdx.z + blockIdx.z*B)*NX*NY;
	qout[ridx] = final; 					// write charge
	qout[ridx + NPTS] = qube[ridx + NPTS]; 	// copy the VNe as it is
}

// this one uses only a 2 component grid for density and VNe
__global__ void __launch_bounds__(512, 4) gpu_automaton_simple_singleshot(
	float 	*qube, 	// input multifield(2) qube
	float 	*qout 	// output multifield(2) qube
	) {


	// can store up to 16 floats per thread in shmem if we use an A100!!!
	volatile __shared__ float buffer1[16*B_3];
	volatile float buffer2[16];

	float final = 0;
	volatile uint ridx, widx;
	volatile short3 r;
	volatile float v1;

	// load V in this point
	// load the data of the thread assigned voxel - 2 inputs
	{
		r.x = threadIdx.x + blockIdx.x*B;
		r.y = threadIdx.y + blockIdx.y*B;
		r.z = threadIdx.z + blockIdx.z*B;
		ridx = r.x + r.y*NX + r.z*NX*NY;
		widx = (threadIdx.x + threadIdx.y*B + threadIdx.z*B_2)*16;

		v1 = qube[ridx + 1*NPTS];
	}

	// this is a loop over neigbours
	for(ushort deltas=1; deltas < 64; deltas++) {

		if((deltas & 3) == 2) deltas++;
		if((deltas & 12) == 8) deltas+=4;
		if((deltas & 48) == 32) deltas+=16;

		volatile float acc;
		volatile float v2;
		
		r.x = threadIdx.x + blockIdx.x*B;
		r.y = threadIdx.y + blockIdx.y*B;
		r.z = threadIdx.z + blockIdx.z*B;

		// load the neigbour point - other 2 inputs
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
			
			v2 = qube[ridx + 1*NPTS];
		}

		buffer1[widx] = v1;
		buffer1[widx+1] = v2;

		// put some extra constants in the last 14 inputs - buffer1[widx + 2 + k] k=0-13(incl)
		// PYCUDA_GENETICPROGRAM_CONSTS_1


		// now write the actual program - NO SYMMETRIES NEEDED?
		// PYCUDA_GENETICPROGRAM_1



		// now buffer2[0] has the transfer factors for the q

		// scale by distance
		float factor = (deltas & 1) + ((deltas & 4)>0) + ((deltas & 16)>0);
		factor = rsqrtf(factor);

		final += buffer2[0]*buffer2[0] * factor;
		__syncthreads();
	}

	// scale it by v1 in the end
	final = v1 * final;
	
	// save the final results
	ridx = (threadIdx.x + blockIdx.x*B) + (threadIdx.y + blockIdx.y*B)*NX + (threadIdx.z + blockIdx.z*B)*NX*NY;
	qout[ridx] = final;
}