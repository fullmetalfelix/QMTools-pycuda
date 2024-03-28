#include "kernel_common.h"

#define NPTS PYCUDA_NPTS
#define NX PYCUDA_NX
#define NY PYCUDA_NY
#define NZ PYCUDA_NZ
#define GRIDSTEP PYCUDA_GRIDSTEP

//#define ADS PYCUDA_ADS
//#define DIF PYCUDA_DIF
#define nf 4


#define OneOverSqrt2 0.7071067811865475
#define OneOverSqrt3 0.5773502691896258
#define OneOverDIFFTOT 0.05234482976098482

#define DIFFQ 0.8
#define DIFFA 0.4
#define DIFFB 0.4

#define DNASIZE PYCUDA_DNASIZE

#define NL1 PYCUDA_NL1
#define NL2 PYCUDA_NL2
#define ABS_A PYCUDA_ABS_A
#define ABS_B PYCUDA_ABS_B


__constant__ float cParams[DNASIZE];


__global__ void __launch_bounds__(512, 4) gpu_automaton_nn_singleshot(
	float 	*qube, 	// input field q/VNe qube (2 fields) --- for simplicity
	float 	*qout 	// output charge qube (1 field1)
	) {

	// can store up to 16 floats per thread in shmem if we use an A100!!!

	volatile __shared__ float buffer1[16*B_3];
	volatile float buffer2[16];

	float final = 0;
	volatile uint ridx, widx;
	volatile float v1;
	volatile short3 r;

	// load the data of the thread assigned voxel - 2 inputs
	{
		r.x = threadIdx.x + blockIdx.x*B;
		r.y = threadIdx.y + blockIdx.y*B;
		r.z = threadIdx.z + blockIdx.z*B;
		ridx = r.x + r.y*NX + r.z*NX*NY;
		widx = (threadIdx.x + threadIdx.y*B + threadIdx.z*B_2)*16;
		
		buffer1[widx] = qube[ridx + 1*NPTS];
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

		// put v1/v2 in the buffer - THIS FUNCTION IS NOT SYMMETRIC NOR ANTISYMMETRIC W.R.T. SWITCHING VOXELS!
		widx = (threadIdx.x + threadIdx.y*B + threadIdx.z*B_2)*16;
		buffer1[widx] = 	v1;
		buffer1[widx + 1] = v2;

		// put some extra constants in the last 12 inputs - buffer1[widx + 4 + k] k=0-11(incl)
		for(ushort k=0; k<14; k++) buffer1[widx + 2 + k] = cParams[k];
		__syncthreads();

		// main NN part
		ushort cnt = 0; acc = 0;
		ushort os = 14;
		while((cnt>>8) != NL1) {

			// (cnt & 15) 		= first 4 bits 	= index of the input [0-15]
			// (cnt >> 4) & 15 	= second 4 bits = index of the layer neuron we are computing
			// (cnt >> 8) 		= third 4 bits (or more!) = index of the layer (up to 256 layers are allowed)

			acc += cParams[os] * buffer1[widx + (cnt & 15)];
			os++;

			if((cnt & 15) == 15) { // true after input 15 (from 0) of the previous layer has been processed
				// add the neuron bias 
				acc += cParams[os];
				os++;
				// compute the neuron output
				buffer2[(cnt>>4)&15] = tanhf(acc);
				acc = 0; // reset the accumulator

				// now we check if this was the last neuron of the layer
				if(((cnt>>4) & 15) == 15) {

					buffer1[widx+0] = buffer2[0]; buffer1[widx+1] = buffer2[1];
					buffer1[widx+2] = buffer2[2]; buffer1[widx+3] = buffer2[3];
					buffer1[widx+4] = buffer2[4]; buffer1[widx+5] = buffer2[5];
					buffer1[widx+6] = buffer2[6]; buffer1[widx+7] = buffer2[7];
					buffer1[widx+8] = buffer2[8]; buffer1[widx+9] = buffer2[9];
					buffer1[widx+10] = buffer2[10]; buffer1[widx+11] = buffer2[11];
					buffer1[widx+12] = buffer2[12]; buffer1[widx+13] = buffer2[13];
					buffer1[widx+14] = buffer2[14]; buffer1[widx+15] = buffer2[15];
				}
			}

			cnt++;
		}

		// last layer - 1 neuron
		cnt = 0;
		while(cnt < 16) {

			acc += cParams[os] * buffer1[widx + cnt];
			os++;	
			cnt++;
		}
		// add the neuron bias - last layer is linear
		acc += cParams[os];
		//acc = tanhf(acc);
		// and this is the transfer factor for q
		// the transfer is T(vox1,vox2) or T(vox2,vox1)
		// tQ is the charge contribution the voxel should have from its neighbour
		acc *= acc; // we square it so it becomes positive!

		// scale by distance
		float factor = (deltas & 1) + ((deltas & 4)>0) + ((deltas & 16)>0);
		factor = rsqrtf(factor)/GRIDSTEP;

		final += acc * factor;
	} // end of loop over neighbours

	// compute write outputs
	final = v1 * final * OneOverDIFFTOT; // this makes the output proportional to the local pseudofield, which is proportional to the charge nearby?

	// save the final results
	// only Q, A and B channels are updated
	ridx = threadIdx.x + blockIdx.x*B + (threadIdx.y + blockIdx.y*B)*NX + (threadIdx.z + blockIdx.z*B)*NX*NY;
	qout[ridx] = final;
}