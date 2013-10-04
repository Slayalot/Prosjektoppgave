/*****************************************************************************/
/*                                                                           */
/*                                                                           */
/* (c) Copyright 2010, 2011, 2012 by                                         */
/*     SINTEF, Oslo, Norway                                                  */
/*     All rights reserved.                                                  */
/*                                                                           */
/*  THIS SOFTWARE IS FURNISHED UNDER A LICENSE AND MAY BE USED AND COPIED    */
/*  ONLY IN  ACCORDANCE WITH  THE  TERMS  OF  SUCH  LICENSE  AND WITH THE    */
/*  INCLUSION OF THE ABOVE COPYRIGHT NOTICE. THIS SOFTWARE OR  ANY  OTHER    */
/*  COPIES THEREOF MAY NOT BE PROVIDED OR OTHERWISE MADE AVAILABLE TO ANY    */
/*  OTHER PERSON.  NO TITLE TO AND OWNERSHIP OF  THE  SOFTWARE IS  HEREBY    */
/*  TRANSFERRED.                                                             */
/*                                                                           */
/*  SINTEF  MAKES NO WARRANTY  OF  ANY KIND WITH REGARD TO THIS SOFTWARE,    */
/*  INCLUDING,   BUT   NOT   LIMITED   TO,  THE  IMPLIED   WARRANTIES  OF    */
/*  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.                    */
/*                                                                           */
/*  SINTEF SHALL NOT BE  LIABLE  FOR  ERRORS  CONTAINED HEREIN OR DIRECT,    */
/*  SPECIAL,  INCIDENTAL  OR  CONSEQUENTIAL  DAMAGES  IN  CONNECTION WITH    */
/*  FURNISHING, PERFORMANCE, OR USE OF THIS MATERIAL.                        */
/*                                                                           */
/*                                                                           */
/*****************************************************************************/

#include "cuda/util.h"
#include "configure.h"

#define USE_1D_INDEX

__constant__ RKKernelArgs rk_ctx;

texture<float, 1, cudaReadModeElementType> texDt;

/**
 * 2nd order Runge-Kutta time integration
 */
template <unsigned int block_width, unsigned int block_height, unsigned int step>
__launch_bounds__(block_width*block_height, 3)
__global__ void RKKernel() {
	const unsigned int nthreads = block_width*block_height;
	int tid = threadIdx.y*blockDim.x+threadIdx.x;
	__shared__ float water[nthreads];
	volatile float* water_volatile = water;
	
	int bx  = blockIdx.x;
	int by  = blockIdx.y;

	int x = bx * blockDim.x + threadIdx.x + 2; // need to add ghost cells for sparse grids, to get a correct per-block indexing
	int y = by * blockDim.y + threadIdx.y + 2; //skip first two rows (they are ghost cells
    
	float h=0.0f, alpha=0.0f;
	float U1, U2, U3;
	double R1, R2, R3;
	float Q1, Q2, Q3;
    float B;
	float n;
	double dt=0.0;

	if (x < 2 || x >= rk_ctx.nx+2 || y >= rk_ctx.ny+2) return;
	
	U1 = device_address2D(rk_ctx.U1.ptr, rk_ctx.U1.pitch, x, y)[0];
	U2 = device_address2D(rk_ctx.U2.ptr, rk_ctx.U2.pitch, x, y)[0];
	U3 = device_address2D(rk_ctx.U3.ptr, rk_ctx.U3.pitch, x, y)[0];

    B = device_address2D(rk_ctx.Bm.ptr, rk_ctx.Bm.pitch, x, y)[0];

	R1 = device_address2D(rk_ctx.R1.ptr, rk_ctx.R1.pitch, x, y)[0];
	R2 = device_address2D(rk_ctx.R2.ptr, rk_ctx.R2.pitch, x, y)[0];
	R3 = device_address2D(rk_ctx.R3.ptr, rk_ctx.R3.pitch, x, y)[0];

	if (rk_ctx.spatially_varying_manning) {
		n = device_address2D(rk_ctx.M.ptr, rk_ctx.M.pitch, x, y)[0];
	}
	else {
		n = device_address2D(rk_ctx.M.ptr, rk_ctx.M.pitch, 0, 0)[0];
	}

	dt = tex1Dfetch(texDt, 0);

	//Calculate manning friction
	h = U1-B;
	if (h > KPSIMULATOR_DRY_EPS) {
		if (n > 0.0f) {
			float hc2 = powf(h, 8.0f/6.0f)/(n*n); //!< hc^2, c=h^(1/6)/n^2 Chezy coefficient.
			float u, v;
			u = U2/h;
			v = U3/h;
			alpha = dt*rk_ctx.g*sqrt(u*u+v*v)/hc2;
		}
	}

	if (step == 0) {
		Q1 =  U1 + dt*R1;
		Q2 = (U2 + dt*R2)/(1.0f+alpha);
		Q3 = (U3 + dt*R3)/(1.0f+alpha);
	}
	else {
		Q1 = device_address2D(rk_ctx.Q1.ptr, rk_ctx.Q1.pitch, x, y)[0];
		Q2 = device_address2D(rk_ctx.Q2.ptr, rk_ctx.Q2.pitch, x, y)[0];
		Q3 = device_address2D(rk_ctx.Q3.ptr, rk_ctx.Q3.pitch, x, y)[0];

		Q1 = 0.5f*(Q1 + (U1 + dt*R1));
		Q2 = 0.5f*(Q2 + (U2 + dt*R2))/(1.0f+0.5f*alpha);
		Q3 = 0.5f*(Q3 + (U3 + dt*R3))/(1.0f+0.5f*alpha);
	}

	h = Q1-B;
	if (h <= KPSIMULATOR_DRY_EPS) { //land
		Q1 = 0.0f;
		Q2 = 0.0f;
		Q3 = 0.0f;
	}

	//Now write out if we have changes in this part...
	//Used in early exit of fgh-kernel
	water[tid] = ((fabs(R1)+fabs(R2)+fabs(R3)) > 0.0f && h > KPSIMULATOR_DRY_EPS && x >= 2 && y >= 2 && x < rk_ctx.nx+2 && y < rk_ctx.ny+2) ? 1.0f : 0.0f;
	__syncthreads();
		
	//First use all threads to reduce max(1024, nthreads) values into 64 values
	//This first outer test is a compile-time test simply to remove statements if nthreads is less than 512.
	if (nthreads >= 512) {
		//This inner test (p < 512) first checks that the current thread should
		//be active in the reduction from min(1024, nthreads) elements to 512. Makes little sense here, but
		//a lot of sense for the last test where there should only be 64 active threads.
		//The second part of this test ((p+512) < nthreads) removes the threads that would generate an
		//out-of-bounds access to shared memory
		if (tid < 512 && (tid+512) < nthreads) water[tid] = max(water[tid], water[tid + 512]); //max(1024, nthreads)=>512
		__syncthreads();
	}
	if (nthreads >= 256) { 
		if (tid < 256 && (tid+256) < nthreads) water[tid] = max(water[tid], water[tid + 256]); //max(512, nthreads)=>256
		__syncthreads();
	}
	if (nthreads >= 128) {
		if (tid < 128 && (tid+128) < nthreads) water[tid] = max(water[tid], water[tid + 128]); //max(256, nthreads)=>128
		__syncthreads();
	}
	if (nthreads >= 64) {
		if (tid < 64 && (tid+64) < nthreads) water[tid] = max(water[tid], water[tid + 64]); //max(128, nthreads)=>64
		__syncthreads();
	}

	//Let the last warp reduce 64 values into a single value
	//Will generate out-of-bounds errors for nthreads < 64
	if (tid < 32) {
		if (nthreads >= 64) water_volatile[tid] = max(water_volatile[tid], water_volatile[tid + 32]); //64=>32
		if (nthreads >= 32) water_volatile[tid] = max(water_volatile[tid], water_volatile[tid + 16]); //32=>16
		if (nthreads >= 16) water_volatile[tid] = max(water_volatile[tid], water_volatile[tid +  8]); //16=>8
		if (nthreads >=  8) water_volatile[tid] = max(water_volatile[tid], water_volatile[tid +  4]); //8=>4
		if (nthreads >=  4) water_volatile[tid] = max(water_volatile[tid], water_volatile[tid +  2]); //4=>2
		if (nthreads >=  2) water_volatile[tid] = max(water_volatile[tid], water_volatile[tid +  1]); //2=>1
	}

	if (tid == 0) device_address2D(rk_ctx.D.ptr, rk_ctx.D.pitch, bx, by)[0] = water_volatile[0];

	device_address2D(rk_ctx.Q1.ptr, rk_ctx.Q1.pitch, x, y)[0] = Q1;
	device_address2D(rk_ctx.Q2.ptr, rk_ctx.Q2.pitch, x, y)[0] = Q2;
	device_address2D(rk_ctx.Q3.ptr, rk_ctx.Q3.pitch, x, y)[0] = Q3;
}










/**
 * 2nd order Runge-Kutta substep kernel
 */
template<unsigned int step>
void RKKernelLauncher(const RKKernelArgs* h_ctx, const KernelConfiguration& config) {
	if(step > 1) {
		std::cout << "This kernel is only valid for 2-step RK" << std::endl;
		exit(-1);
	}

	//Upload parameters to the GPU
	KPSIMULATOR_CHECK_CUDA(cudaMemcpyToSymbolAsync(rk_ctx, h_ctx, sizeof(RKKernelArgs), 0, cudaMemcpyHostToDevice, config.stream));
	cudaChannelFormatDesc dt_texdesc = cudaCreateChannelDesc<float>();
	texDt.addressMode[0] = cudaAddressModeClamp;
	texDt.addressMode[1] = cudaAddressModeClamp;
	texDt.filterMode = cudaFilterModePoint;
	texDt.normalized = false;
	KPSIMULATOR_CHECK_CUDA(cudaBindTexture(NULL, &texDt, h_ctx->dt, &dt_texdesc, sizeof(float)));

	//Launch kernel
	cudaFuncSetCacheConfig(RKKernel<KPSIMULATOR_RK_BLOCK_WIDTH, KPSIMULATOR_RK_BLOCK_HEIGHT, step>, cudaFuncCachePreferL1);
	RKKernel<KPSIMULATOR_RK_BLOCK_WIDTH, KPSIMULATOR_RK_BLOCK_HEIGHT, step><<<config.grid, config.block, 0, config.stream>>>();
	KPSIMULATOR_CHECK_CUDA_ERROR("RKKernel");
}
