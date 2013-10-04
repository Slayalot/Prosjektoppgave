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

#ifndef INITU_HPP_
#define INITU_HPP_

#include "cuda/util.h"

template <unsigned int block_width, unsigned int block_height>
__global__ void initUKernel(gpu_raw_ptr<> U1, gpu_raw_ptr<> U2, gpu_raw_ptr<> U3, gpu_raw_ptr<> Bm,
		unsigned int width, unsigned int height) {
	float* U1_ptr;
	float* U2_ptr;
	float* U3_ptr;
	float* Bm_ptr;
	float w, B, h;

	unsigned int bx0 = blockIdx.x*blockDim.x;
	unsigned int by0 = blockIdx.y*blockDim.y;
	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;
	unsigned int out_x = bx0+tx;
	unsigned int out_y = by0+ty;

	U1_ptr = device_address2D(U1.ptr, U1.pitch, out_x, out_y);
	U2_ptr = device_address2D(U2.ptr, U2.pitch, out_x, out_y);
	U3_ptr = device_address2D(U3.ptr, U3.pitch, out_x, out_y);
	Bm_ptr = device_address2D(Bm.ptr, Bm.pitch, out_x, out_y);

	if (out_x < width && out_y < height) {
		w = U1_ptr[0];
		B = Bm_ptr[0];
		h = w - B;
		if (h <= 0.0f) {
			U1_ptr[0] = B;
			U2_ptr[0] = 0.0f;
			U3_ptr[0] = 0.0f;
		}
	}
}

#endif /* INITU_HPP_ */
