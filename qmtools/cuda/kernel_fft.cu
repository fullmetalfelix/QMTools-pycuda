#include "kernel_common.h"
#include <cuComplex.h>

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

#define R_BOHR 0.529177211 	// Bohr-radius in angstroms
#define EPS0 0.0055263494 	// e/(V*Ang), vacuum permittivity

#include <stdio.h>

// Add nuclear charge to electron density grid and scale with vacuum permittivity
__global__ void gpu_add_nuclear_charge(
    float* 		q_in,		// input electron density grid
    float*  	q_out,		// output total charge density grid
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
        scoords[idx].x = tmp.x * R_BOHR;
        scoords[idx].y = tmp.y * R_BOHR;
        scoords[idx].z = tmp.z * R_BOHR;
    }

    __syncthreads();

    // Grid position index
    int i = blockIdx.x * B + threadIdx.x;
    int j = blockIdx.y * B + threadIdx.y;
    int k = blockIdx.z * B + threadIdx.z;

    // Real-space position of voxel
    float3 pos;
    pos.x = X0 + (i + 0.5) * STEP;
    pos.y = Y0 + (j + 0.5) * STEP;
    pos.z = Z0 + (k + 0.5) * STEP;

    // Get electron density (in fortran order for some reason)
    float q = q_in[i + j * NX + k * NX * NY];
    q /= -STEP*STEP*STEP; // Convert to electron density in -e/ang^3

    // Loop over atoms
    float c, r2;
    for (ushort i=0; i<NATM; i++) {

        // Squared distance from voxel to atom
        c = pos.x - scoords[i].x; r2 = c*c;
        c = pos.y - scoords[i].y; r2+= c*c;
        c = pos.z - scoords[i].z; r2+= c*c;

        // Add to voxel if within cutoff
        if (r2 <= CUTOFF_SQ) {
            q += styp[i] / (SIGMA*SIGMA*SIGMA*2*M_PI*sqrtf(2*M_PI)) * expf(-r2/(2*SIGMA*SIGMA));
        }

    }

    // Write to output array
    q_out[i*NY*NZ + j*NZ + k] = -q / EPS0;

}

// Solve poisson equation in frequency space
__global__ void gpu_poisson_frequency_solve(
    cuComplex* rho_hat // Fourier-transformed right-hand-side of the poisson equation
) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i == 0 && j == 0 && k == 0) {
        rho_hat[0].x = 0.0;
        rho_hat[0].y = 0.0;
    } else {
        
        // Get frequency components
        float kx = i > (NX/2 - 1) ? (float) (i - NX) / (NX * STEP) : (float) i / (NX * STEP);
        float ky = j > (NY/2 - 1) ? (float) (j - NY) / (NY * STEP) : (float) j / (NY * STEP);
        float kz = k > (NZ/2 - 1) ? (float) (k - NZ) / (NZ * STEP) : (float) k / (NZ * STEP);

        float scale = -1/(4.0*M_PI*M_PI * (kx*kx + ky*ky + kz*kz));
        int idx = i*NY*NZ + j*NZ + k;
        rho_hat[idx].x *= scale;
        rho_hat[idx].y *= scale;

    }

}
