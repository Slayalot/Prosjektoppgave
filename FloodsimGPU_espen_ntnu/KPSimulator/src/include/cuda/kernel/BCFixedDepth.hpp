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

#ifndef BCFIXEDDEPTH_HPP_
#define BCFIXEDDEPTH_HPP_

#include "cuda/util.h"

struct BCFixedDepthNorth {
	template<unsigned int threads>
	__device__ static inline void set(gpu_raw_ptr<> U1, gpu_raw_ptr<> U2, gpu_raw_ptr<> U3,
			const unsigned int& w, const unsigned int& h,
			const unsigned int& tid) {
		float* U_out[3];
		float* U_in[3];

		//Set first ghost cell to the wanted height
		U_out[0] = device_address2D(U1.ptr, U1.pitch, 0, h-2);
		U_out[1] = device_address2D(U2.ptr, U2.pitch, 0, h-2);
		U_out[2] = device_address2D(U3.ptr, U3.pitch, 0, h-2);

		U_in[0] = device_address2D(U1.ptr, U1.pitch, 0, h-3);
		U_in[1] = device_address2D(U2.ptr, U2.pitch, 0, h-3);
		U_in[2] = device_address2D(U3.ptr, U3.pitch, 0, h-3);
		for (unsigned int i=tid+2; i<w-2; i+=threads) {
			U_out[0][i] = bc_ctx.north_arg;
			U_out[1][i] = U_in[1][i];
			U_out[2][i] = U_in[2][i];
		}

		//Then force minmod to return 0 slope.
		U_out[0] = device_address2D(U1.ptr, U1.pitch, 0, h-1);
		U_out[1] = device_address2D(U2.ptr, U2.pitch, 0, h-1);
		U_out[2] = device_address2D(U3.ptr, U3.pitch, 0, h-1);

		U_in[0] = device_address2D(U1.ptr, U1.pitch, 0, h-3);
		U_in[1] = device_address2D(U2.ptr, U2.pitch, 0, h-3);
		U_in[2] = device_address2D(U3.ptr, U3.pitch, 0, h-3);
		for (unsigned int i=tid+2; i<w-2; i+=threads) {
			U_out[0][i] = bc_ctx.north_arg;
			U_out[1][i] = U_in[1][i];
			U_out[2][i] = U_in[2][i];
		}
	}
};

struct BCFixedDepthSouth {
	template<unsigned int threads>
	__device__ static inline void set(gpu_raw_ptr<> U1, gpu_raw_ptr<> U2, gpu_raw_ptr<> U3,
			const unsigned int& w, const unsigned int& h,
			const unsigned int& tid) {
		float* U_out[3];
		float* U_in[3];

		U_out[0] = device_address2D(U1.ptr, U1.pitch, 0, 1);
		U_out[1] = device_address2D(U2.ptr, U2.pitch, 0, 1);
		U_out[2] = device_address2D(U3.ptr, U3.pitch, 0, 1);

		U_in[0] = device_address2D(U1.ptr, U1.pitch, 0, 2);
		U_in[1] = device_address2D(U2.ptr, U2.pitch, 0, 2);
		U_in[2] = device_address2D(U3.ptr, U3.pitch, 0, 2);
		for (unsigned int i=tid+2; i<w-2; i+=threads) {
			U_out[0][i] = bc_ctx.south_arg;
			U_out[1][i] = U_in[1][i];
			U_out[2][i] = U_in[2][i];
		}

		U_out[0] = device_address2D(U1.ptr, U1.pitch, 0, 0);
		U_out[1] = device_address2D(U2.ptr, U2.pitch, 0, 0);
		U_out[2] = device_address2D(U3.ptr, U3.pitch, 0, 0);

		U_in[0] = device_address2D(U1.ptr, U1.pitch, 0, 2);
		U_in[1] = device_address2D(U2.ptr, U2.pitch, 0, 2);
		U_in[2] = device_address2D(U3.ptr, U3.pitch, 0, 2);
		for (unsigned int i=tid+2; i<w-2; i+=threads) {
			U_out[0][i] = bc_ctx.south_arg;
			U_out[1][i] = U_in[1][i];
			U_out[2][i] = U_in[2][i];
		}
	}
};

struct BCFixedDepthEast {
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
				U_out[0][(w-1) - i] = bc_ctx.east_arg;
				U_out[1][(w-1) - i] = device_address2D(U2.ptr, U2.pitch, w-3, j)[0];
				U_out[2][(w-1) - i] = device_address2D(U3.ptr, U3.pitch, w-3, j)[0];
			}
		}
	}
};

struct BCFixedDepthWest {
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
				U_out[0][i] = bc_ctx.west_arg;
				U_out[1][i] = device_address2D(U2.ptr, U2.pitch, 2, j)[0];
				U_out[2][i] = device_address2D(U3.ptr, U3.pitch, 2, j)[0];
			}
		}
	}
};


#endif /* BCFIXEDDEPTH_HPP_ */
