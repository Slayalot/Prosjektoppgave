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

#ifndef KPSIMULATORHELPERS_HPP_
#define KPSIMULATORHELPERS_HPP_

#include "configure.h"


/**
 * Calculates the minimum grid size needed to cover the full width*height domain
 * @param block_size
 * @param width
 * @param height
 * @return the grid size
 */
inline dim3 get_grid_dim(dim3 block_size, unsigned int width, unsigned int height) {
	dim3 retval((width + (block_size.x - 1)) / block_size.x,
		        (height + (block_size.y - 1)) / block_size.y);
	return retval;
}

inline KernelConfiguration get_rk_config(unsigned int width, unsigned int height, cudaStream_t stream) {
	KernelConfiguration config;

	if (width == 0 || height == 0) {
		std::cout << "Wrong number of blocks for rk" << std::endl;
		std::cout << "domain=" << width << "x" << height << std::endl;
		exit(-1);
	}

	config.block = dim3(KPSIMULATOR_RK_BLOCK_WIDTH, KPSIMULATOR_RK_BLOCK_HEIGHT);
	config.grid = get_grid_dim(config.block, width+2, height+2);
	config.stream = stream;

#ifndef NDEBUG
	std::cout << "RK " << config << std::endl;
#endif

	return config;
}


inline KernelConfiguration get_flux_config(unsigned int width, unsigned int height, cudaStream_t stream) {
	KernelConfiguration config;

	if (width == 0 || height == 0) {
		std::cout << "Wrong number of blocks for flux" << std::endl;
		std::cout << "domain=" << width << "x" << height << std::endl;
		exit(-1);
	}

	config.block = dim3(KPSIMULATOR_FLUX_BLOCK_WIDTH, KPSIMULATOR_FLUX_BLOCK_HEIGHT);
	config.grid = get_grid_dim(config.block, width+2, height+2); //!< XXX: Added two here to make flux-grid==rk-grid, to make grow-kernel work properly. Is this the correct fix?
	config.stream = stream;

#ifndef NDEBUG
	std::cout << "Flux " << config << std::endl;
#endif

	return config;
}

inline KernelConfiguration get_dt_config(unsigned int elements, cudaStream_t stream) {
	KernelConfiguration config;

	if (elements == 0) {
		std::cout << "Wrong number of elements for dt" << std::endl;
		std::cout << "elements=" << elements << std::endl;
		exit(-1);
	}

	//Set the grid and block size
	config.stream = stream;
	config.grid = dim3(1, 1, 1);
	config.block = dim3(1, 1, 1);
	config.block.x = (elements >=   2) ?   2 : config.block.x;
	config.block.x = (elements >=   4) ?   4 : config.block.x;
	config.block.x = (elements >=   8) ?   8 : config.block.x;
	config.block.x = (elements >=  16) ?  16 : config.block.x;
	config.block.x = (elements >=  32) ?  32 : config.block.x;
	config.block.x = (elements >=  64) ?  64 : config.block.x;
	config.block.x = (elements >= 128) ? 128 : config.block.x;
	config.block.x = (elements >= 256) ? 256 : config.block.x;
	config.block.x = (elements >= 512) ? 512 : config.block.x;


#ifndef NDEBUG
	std::cout << "Dt " << config << std::endl;
#endif

	return config;
}

inline KernelConfiguration get_bc_config(unsigned int elements, cudaStream_t stream) {
	KernelConfiguration config;

	//Set the grid and block size
	config.stream = stream;
	config.grid = dim3(1, 1, 1);
#ifdef KPSIMULATOR_USE_BC_SMALL_DOMAINS
	config.block = dim3(1, 1, 1);
	config.block.x = (elements >=   2) ?   2 : config.block.x;
	config.block.x = (elements >=   4) ?   4 : config.block.x;
	config.block.x = (elements >=   8) ?   8 : config.block.x;
	config.block.x = (elements >=  16) ?  16 : config.block.x;
	config.block.x = (elements >=  32) ?  32 : config.block.x;
	config.block.x = (elements >=  64) ?  64 : config.block.x;
	config.block.x = (elements >= 128) ? 128 : config.block.x;
	config.block.x = (elements >= 256) ? 256 : config.block.x;
#else
	config.block = dim3(256, 1, 1);
#endif
#if __CUDA_ARCH__ >= 200
	config.block.x = (elements >= 512) ? 512 : config.block.x;
#endif


#ifndef NDEBUG
	std::cout << "BC " << config << std::endl;
#endif

	return config;
}

inline void set_flux_args(FGHKernelArgs* args,
		gpu_raw_ptr<> U1, gpu_raw_ptr<> U2, gpu_raw_ptr<> U3,
		gpu_raw_ptr<> R1, gpu_raw_ptr<> R2, gpu_raw_ptr<> R3,
		gpu_raw_ptr<> D,
		gpu_raw_ptr<> Bi,
		gpu_raw_ptr<> Bm,
		float* L,
		float dx, float dy,
		unsigned int nx, unsigned int ny,
		float g) {
	args->U1 = U1;
	args->U2 = U2;
	args->U3 = U3;

	args->R1 = R1;
	args->R2 = R2;
	args->R3 = R3;
	args->D  = D;
	args->L = L;
	args->Bi = Bi;
	args->Bm = Bm;
	args->dx = dx;
	args->dy = dy;
	args->nx = nx;
	args->ny = ny;
	args->g = g;
}

inline void set_rk_args(RKKernelArgs* args,
		gpu_raw_ptr<> U1, gpu_raw_ptr<> U2, gpu_raw_ptr<> U3,
		gpu_raw_ptr<> Q1, gpu_raw_ptr<> Q2, gpu_raw_ptr<> Q3,
		gpu_raw_ptr<> R1, gpu_raw_ptr<> R2, gpu_raw_ptr<> R3,
		gpu_raw_ptr<> M,
		gpu_raw_ptr<> D,
		gpu_raw_ptr<> Bm,
		float* dt_d,
		unsigned int nx, unsigned int ny,
		float g,
		bool spatially_varying_manning) {
	args->U1 = U1;
	args->U2 = U2;
	args->U3 = U3;

	args->Q1 = Q1;
	args->Q2 = Q2;
	args->Q3 = Q3;

	args->R1 = R1;
	args->R2 = R2;
	args->R3 = R3;
	args->M = M;
	args->D = D;
	args->Bm = Bm;
	args->dt = dt_d;
	args->g = g;
	args->nx = nx;
	args->ny = ny;

}

inline void set_bc_args(BCKernelArgs* args,
		gpu_raw_ptr<> U1, gpu_raw_ptr<> U2, gpu_raw_ptr<> U3,
		unsigned int nx, unsigned int ny) {
	args->U1 = U1;
	args->U2 = U2;
	args->U3 = U3;

	args->width = nx;
	args->height = ny;
}

inline void set_dt_args(DtKernelArgs* args,
		float* L, unsigned int elements, float* dt,
		float dx, float dy, float scale) {
	args->L        = L;
	args->elements = elements;
	args->dt       = dt;
	args->dx       = dx;
	args->dy       = dy;
	args->scale    = scale;

}

#endif /* KPSIMULATORHELPERS_HPP_ */
