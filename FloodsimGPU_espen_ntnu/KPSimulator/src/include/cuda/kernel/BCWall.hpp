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

#ifndef BCWALL_HPP_
#define BCWALL_HPP_

#include "cuda/util.h"

struct BCWallNorth {
	template<unsigned int threads>
	__device__ static inline void set(gpu_raw_ptr<> U1, gpu_raw_ptr<> U2, gpu_raw_ptr<> U3,
			const unsigned int& w, const unsigned int& h,
			const unsigned int& tid) {
		float* U_out[3];
		float* U_in[3];
		for (unsigned int j=0; j<2; ++j) {
			U_out[0] = device_address2D(U1.ptr, U1.pitch, 0, (h-1) - j);
			U_out[1] = device_address2D(U2.ptr, U2.pitch, 0, (h-1) - j);
			U_out[2] = device_address2D(U3.ptr, U3.pitch, 0, (h-1) - j);

			U_in[0] = device_address2D(U1.ptr, U1.pitch, 0, (h-1) - 3 + j);
			U_in[1] = device_address2D(U2.ptr, U2.pitch, 0, (h-1) - 3 + j);
			U_in[2] = device_address2D(U3.ptr, U3.pitch, 0, (h-1) - 3 + j);
			for (unsigned int i=tid+2; i<w-2; i+=threads) {
				U_out[0][i] =  U_in[0][i];
				U_out[1][i] =  U_in[1][i];
				U_out[2][i] = -U_in[2][i];
			}
		}
	}
};

struct BCWallSouth {
	template<unsigned int threads>
	__device__ static inline void set(gpu_raw_ptr<> U1, gpu_raw_ptr<> U2, gpu_raw_ptr<> U3,
			const unsigned int& w, const unsigned int& h,
			const unsigned int& tid) {
		float* U_out[3];
		float* U_in[3];
		for (unsigned int j = 0; j < 2; ++j) {
			U_out[0] = device_address2D(U1.ptr, U1.pitch, 0, j);
			U_out[1] = device_address2D(U2.ptr, U2.pitch, 0, j);
			U_out[2] = device_address2D(U3.ptr, U3.pitch, 0, j);

			U_in[0] = device_address2D(U1.ptr, U1.pitch, 0, 3 - j);
			U_in[1] = device_address2D(U2.ptr, U2.pitch, 0, 3 - j);
			U_in[2] = device_address2D(U3.ptr, U3.pitch, 0, 3 - j);
			for (unsigned int i=tid+2; i<w-2; i+=threads) {
				U_out[0][i] =  U_in[0][i];
				U_out[1][i] =  U_in[1][i];
				U_out[2][i] = -U_in[2][i];
			}
		}
	}
};

struct BCWallEast {
	template<unsigned int threads>
	__device__ static inline void set(gpu_raw_ptr<> U1, gpu_raw_ptr<> U2, gpu_raw_ptr<> U3,
			const unsigned int& w, const unsigned int& h,
			const unsigned int& tid) {
		float* U_out[3];
		for (unsigned int j=tid+2; j<h-2; j+=threads) {
			U_out[0] = device_address2D(U1.ptr, U1.pitch, 0, j);
			U_out[1] = device_address2D(U2.ptr, U2.pitch, 0, j);
			U_out[2] = device_address2D(U3.ptr, U3.pitch, 0, j);
			for (unsigned int i=0; i<2; ++i) {
				U_out[0][(w-1) - i] =  device_address2D(U1.ptr, U1.pitch, (w-1) - 3 + i, j)[0];
				U_out[1][(w-1) - i] = -device_address2D(U2.ptr, U2.pitch, (w-1) - 3 + i, j)[0];
				U_out[2][(w-1) - i] =  device_address2D(U3.ptr, U3.pitch, (w-1) - 3 + i, j)[0];
			}
		}
	}
};

struct BCWallWest {
	template<unsigned int threads>
	__device__ static inline void set(gpu_raw_ptr<> U1, gpu_raw_ptr<> U2, gpu_raw_ptr<> U3,
			const unsigned int& w, const unsigned int& h,
			const unsigned int& tid) {
		float* U_out[3];
		for (unsigned int j=tid+2; j<h-2; j+=threads) {
			U_out[0] = device_address2D(U1.ptr, U1.pitch, 0, j);
			U_out[1] = device_address2D(U2.ptr, U2.pitch, 0, j);
			U_out[2] = device_address2D(U3.ptr, U3.pitch, 0, j);
			for (unsigned int i=0; i<2; ++i) {
				U_out[0][i] =  device_address2D(U1.ptr, U1.pitch, 3 - i, j)[0];
				U_out[1][i] = -device_address2D(U2.ptr, U2.pitch, 3 - i, j)[0];
				U_out[2][i] =  device_address2D(U3.ptr, U3.pitch, 3 - i, j)[0];
			}
		}
	}
};


#endif /* BCWALL_HPP_ */
