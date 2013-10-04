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

#include <float.h>
#include "cuda/util.h"

__constant__ DtKernelArgs dt_ctx;

template <unsigned int threads>
__launch_bounds__(threads, 1)
__global__ void DtKernel() {
	__shared__ float sdata[threads];
	volatile float* sdata_volatile = sdata;
	unsigned int tid = threadIdx.x;
	float dt;

	//Reduce to "threads" elements
	sdata[tid] = FLT_MAX;
	for (unsigned int i=tid; i<dt_ctx.elements; i += threads)
		sdata[tid] = min(sdata[tid], dt_ctx.L[i]);
	__syncthreads();

	//Now, reduce all elements into a single element
	if (threads >= 512) {
		if (tid < 256) sdata[tid] = min(sdata[tid], sdata[tid + 256]);
		__syncthreads();
	}
	if (threads >= 256) {
		if (tid < 128) sdata[tid] = min(sdata[tid], sdata[tid + 128]);
		__syncthreads();
	}
	if (threads >= 128) {
		if (tid < 64) sdata[tid] = min(sdata[tid], sdata[tid + 64]);
		__syncthreads();
	}
	if (tid < 32) {
		if (threads >= 64) sdata_volatile[tid] = min(sdata_volatile[tid], sdata_volatile[tid + 32]);
		if (tid < 16) {
			if (threads >= 32) sdata_volatile[tid] = min(sdata_volatile[tid], sdata_volatile[tid + 16]);
			if (threads >= 16) sdata_volatile[tid] = min(sdata_volatile[tid], sdata_volatile[tid +  8]);
			if (threads >=  8) sdata_volatile[tid] = min(sdata_volatile[tid], sdata_volatile[tid +  4]);
			if (threads >=  4) sdata_volatile[tid] = min(sdata_volatile[tid], sdata_volatile[tid +  2]);
			if (threads >=  2) sdata_volatile[tid] = min(sdata_volatile[tid], sdata_volatile[tid +  1]);
		}

		if (tid == 0) {
			dt = sdata_volatile[tid];
			if (dt == FLT_MAX) {
				//If no water at all, and no sources, 
				//we really do not need to simulate, 
				//but using FLT_MAX will make things crash...
				dt = 1.0e-7f;
			}
			dt_ctx.dt[tid] = dt*dt_ctx.scale;
		}
	}
}





template <unsigned int threads>
inline void dtKernelLauncher(const KernelConfiguration& config) {
	cudaFuncSetCacheConfig(DtKernel<threads>, cudaFuncCachePreferL1);
	DtKernel<threads><<<config.grid, config.block, 0, config.stream>>>();
	KPSIMULATOR_CHECK_CUDA_ERROR("DtKernel");
}
