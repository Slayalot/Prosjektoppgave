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

#ifndef INITBM_HPP_
#define INITBM_HPP_

#include "cuda/util.h"

template <unsigned int block_width, unsigned int block_height>
__global__ void initBmKernel(gpu_raw_ptr<> Bm, gpu_raw_ptr<> Bi,
		unsigned int width, unsigned int height) {
	const unsigned int sm_dim_x = block_width+1;
	const unsigned int sm_dim_y = block_height+1;
	__shared__ float smem[sm_dim_x][sm_dim_y];

	float* src;
	float* dst;

	unsigned int bx0 = blockIdx.x*blockDim.x;
	unsigned int by0 = blockIdx.y*blockDim.y;
	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;

	for (int j=threadIdx.y; j<sm_dim_y; j+=block_height) {
		src = device_address2D(Bi.ptr, Bi.pitch, bx0, min(by0+j, height-1));
		for (int i=threadIdx.x; i<sm_dim_x; i+=block_width) {
			smem[j][i] = src[min(i, width-1)];
		}
	}
	__syncthreads();

	if (bx0+tx < width && by0+ty < height) {
		dst = device_address2D(Bm.ptr, Bm.pitch, bx0+tx, by0+ty);
		dst[0] = 0.25f*(smem[ty][tx] + smem[ty+1][tx] + smem[ty][tx+1] + smem[ty+1][tx+1]);
	}
}

#endif /* INITBM_HPP_ */
