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

#include <limits>
#include <sstream>
#include <stdexcept>

#include "cuda/util.h"

/// kernel parameters
__constant__ BCKernelArgs bc_ctx;

#include "gpu_raw_ptr.hpp"
#include "configure.h"
#include "cuda/kernel/BCWall.hpp"
#include "cuda/kernel/BCFixedDepth.hpp"
#include "cuda/kernel/BCFixedDischarge.hpp"
#include "cuda/kernel/BCOpen.hpp"


template <unsigned int threads>
__launch_bounds__(threads, 1)
__global__ void BCWallSetupKernel(gpu_raw_ptr<float> Bi, gpu_raw_ptr<float> Bm, unsigned int width, unsigned int height) {
	int tid = threadIdx.y*blockDim.x+threadIdx.x; //!< thread id

	float* B_in;
	float* B_out;

	unsigned int w = width+4;
	unsigned int h = height+4;

	/**
	 * south boundary at intersections
	 */
	for (unsigned int j=0; j<2; ++j) {
		B_out = device_address2D(Bi.ptr, Bi.pitch, 0, j);
		B_in  = device_address2D(Bi.ptr, Bi.pitch, 0, 4 - j);
		for (unsigned int i=tid+2; i<w-1; i += threads) //<= w for visual purposes (fix the corner node.
			B_out[i] = B_in[i];
	}

    /** 
      * south boundary at midpoints
      */
    for (unsigned int j=0; j<2; ++j) {
        B_out = device_address2D(Bm.ptr, Bm.pitch, 0, j);
        B_in  = device_address2D(Bm.ptr, Bm.pitch, 0, 3 - j);
        for (unsigned int i=tid+2; i<w-2; i += threads)
            B_out[i] = B_in[i];
    }

	/**
	 * north boundary at intersections
	 */
	for (unsigned int j=0; j<2; ++j) {
		B_out = device_address2D(Bi.ptr, Bi.pitch, 0, h - j);
		B_in  = device_address2D(Bi.ptr, Bi.pitch, 0, h - 4 + j);
		for (unsigned int i=tid+2; i<w-1; i += threads)
			B_out[i] = B_in[i];
	}

    /**
      * North boundary at midpoints
      */
    for (unsigned int j=0; j<2; ++j) {
        B_out = device_address2D(Bm.ptr, Bm.pitch, 0, h-1 - j);
        B_in  = device_address2D(Bm.ptr, Bm.pitch, 0, h-1 - 3 + j);
        for (unsigned int i=tid+2; i<w-2; i += threads)
            B_out[i] = B_in[i];
    }
    
	/**
	 * west boundary at intersections
	 */
	for (unsigned int j=tid+2; j<h-1; j += threads) {
		B_out = device_address2D(Bi.ptr, Bi.pitch, 0, j);
		for (unsigned int i=0; i<2; ++i)
			B_out[i] = device_address2D(Bi.ptr, Bi.pitch, 4 - i, j)[0];
	}

	/**
	 * west boundary at midpoints
	 */
	for (unsigned int j=tid+2; j<h-2; j += threads) {
		B_out = device_address2D(Bm.ptr, Bm.pitch, 0, j);
		for (unsigned int i=0; i<2; ++i)
			B_out[i] = device_address2D(Bm.ptr, Bm.pitch, 3 - i, j)[0];
	}

	/**
	 * east boundary at intersections
	 */
	for (unsigned int j=tid+2; j<h-1; j += threads) {
		B_out = device_address2D(Bi.ptr, Bi.pitch, 0, j);
		for (unsigned int i=0; i<2; ++i)
			B_out[w - i] = device_address2D(Bi.ptr, Bi.pitch, w - 4 + i, j)[0];
	}

	/**
	 * east boundary at midpoints
	 */
	for (unsigned int j=tid+2; j<h-2; j += threads) {
		B_out = device_address2D(Bm.ptr, Bm.pitch, 0, j);
		for (unsigned int i=0; i<2; ++i)
			B_out[w-1 - i] = device_address2D(Bm.ptr, Bm.pitch, w-1 - 3 + i, j)[0];
	}
}



//http://herbsutter.com/2009/10/18/mailbag-shutting-up-compiler-warnings/
template<class T> __device__ void ignore( const T& ) { }
template <unsigned int threads, class N, class S, class E, class W>
__launch_bounds__(threads, 1)
__global__ void BCKernel() {
	int tid = threadIdx.y*blockDim.x+threadIdx.x; //!< thread id
	unsigned int w = bc_ctx.width+4;
	unsigned int h = bc_ctx.height+4;

	N::set<threads>(bc_ctx.U1, bc_ctx.U2, bc_ctx.U3, w, h, tid);
	S::set<threads>(bc_ctx.U1, bc_ctx.U2, bc_ctx.U3, w, h, tid);
	E::set<threads>(bc_ctx.U1, bc_ctx.U2, bc_ctx.U3, w, h, tid);
	W::set<threads>(bc_ctx.U1, bc_ctx.U2, bc_ctx.U3, w, h, tid);

	ignore(tid);
	ignore(w);
	ignore(h);
}











/**
 * Do not perform any boundary conditions
 */
struct BCNone {
	template<unsigned int threads>
	__device__ static inline void set(gpu_raw_ptr<>& U1, gpu_raw_ptr<>& U2, gpu_raw_ptr<>& U3,
			const unsigned int& w, const unsigned int& h,
			const unsigned int& tid) {}
};

template <unsigned int threads, class N, class S, class E, class W>
inline void BCKernelLauncher(const KernelConfiguration& config) {
	cudaFuncSetCacheConfig(BCKernel<threads, N, S, E, W>, cudaFuncCachePreferL1);
	BCKernel<threads, N, S, E, W><<<config.grid, config.block, 0, config.stream>>>();
	KPSIMULATOR_CHECK_CUDA_ERROR("BCKernel");
}


/**
 * Kernel launcher that launches the correct boundary conditions
 * as defined by the template parameters.
 * @param step RK-integration step
 * @param N North boundary condition
 * @param S South boundary condition
 * @param E East boundary condition
 * @param W West boundary condition
 */
template <class N, class S, class E, class W>
inline void BCKernelLauncher(const KernelConfiguration& config) {
	switch(config.block.x) {
#if __CUDA_ARCH__ >= 200
		case 512: BCKernelLauncher<512, N, S, E, W>(config); break;
#endif
		case 256: BCKernelLauncher<256, N, S, E, W>(config); break;
#ifdef KPSIMULATOR_USE_BC_SMALL_DOMAINS
		case 128: BCKernelLauncher<128, N, S, E, W>(config); break;
		case  64: BCKernelLauncher< 64, N, S, E, W>(config); break;
		case  32: BCKernelLauncher< 32, N, S, E, W>(config); break;
		case  16: BCKernelLauncher< 16, N, S, E, W>(config); break;
		case   8: BCKernelLauncher<  8, N, S, E, W>(config); break;
		case   4: BCKernelLauncher<  4, N, S, E, W>(config); break;
		case   2: BCKernelLauncher<  2, N, S, E, W>(config); break;
		case   1: BCKernelLauncher<  1, N, S, E, W>(config); break;
#endif
	}
	KPSIMULATOR_CHECK_CUDA_ERROR("BCKernel");
}


/**
 * Used to translate runtime-parameter west to compile-time kernel for west boundary
 */
template <class N, class S, class E>
inline void BCKernelLauncher(const KernelConfiguration& config,
		KPBoundaryCondition::TYPE west) {
	switch (west) {
	case KPBoundaryCondition::NONE:            BCKernelLauncher<N, S, E, BCNone>(config); break;
#ifdef KPSIMULATOR_USE_BC_WALL_W
	case KPBoundaryCondition::WALL:            BCKernelLauncher<N, S, E, BCWallWest>(config); break;
#endif
#ifdef KPSIMULATOR_USE_BC_FIXED_DEPTH_W
	case KPBoundaryCondition::FIXED_DEPTH:     BCKernelLauncher<N, S, E, BCFixedDepthWest>(config); break;
#endif
#ifdef KPSIMULATOR_USE_BC_FIXED_DISCHARGE_W
	case KPBoundaryCondition::FIXED_DISCHARGE: BCKernelLauncher<N, S, E, BCFixedDischargeWest>(config); break;
#endif
#ifdef KPSIMULATOR_USE_BC_OPEN_W
	case KPBoundaryCondition::OPEN:            BCKernelLauncher<N, S, E, BCOpenWest>(config); break;
#endif
	default:
		std::stringstream log;
		log << __FILE__ << ":" << __LINE__ << " Boundary condition not compiled." << std::endl;
		throw std::runtime_error(log.str());
	}
}

/**
 * Used to translate runtime-parameter east to compile-time kernel for east boundary
 */
template <class N, class S>
inline void BCKernelLauncher(const KernelConfiguration& config,
		KPBoundaryCondition::TYPE east,
		KPBoundaryCondition::TYPE west) {
	switch (east) {
	case KPBoundaryCondition::NONE:            BCKernelLauncher<N, S, BCNone>(config, west); break;
#ifdef KPSIMULATOR_USE_BC_WALL_E
	case KPBoundaryCondition::WALL:            BCKernelLauncher<N, S, BCWallEast>(config, west); break;
#endif
#ifdef KPSIMULATOR_USE_BC_FIXED_DEPTH_E
	case KPBoundaryCondition::FIXED_DEPTH:     BCKernelLauncher<N, S, BCFixedDepthEast>(config, west); break;
#endif
#ifdef KPSIMULATOR_USE_BC_FIXED_DISCHARGE_E
	case KPBoundaryCondition::FIXED_DISCHARGE: BCKernelLauncher<N, S, BCFixedDischargeEast>(config, west); break;
#endif
#ifdef KPSIMULATOR_USE_BC_OPEN_E
	case KPBoundaryCondition::OPEN:            BCKernelLauncher<N, S, BCOpenEast>(config, west); break;
#endif
	default:
		std::stringstream log;
		log << __FILE__ << ":" << __LINE__ << " Boundary condition not compiled." << std::endl;
		throw std::runtime_error(log.str());
	}
}

/**
 * Used to translate runtime-parameter south to compile-time kernel for south boundary
 */
template <class N>
inline void BCKernelLauncher(const KernelConfiguration& config,
		KPBoundaryCondition::TYPE south,
		KPBoundaryCondition::TYPE east,
		KPBoundaryCondition::TYPE west) {
	switch (south) {
	case KPBoundaryCondition::NONE:            BCKernelLauncher<N, BCNone>(config, east, west); break;
#ifdef KPSIMULATOR_USE_BC_WALL_S
	case KPBoundaryCondition::WALL:            BCKernelLauncher<N, BCWallSouth>(config, east, west); break;
#endif
#ifdef KPSIMULATOR_USE_BC_FIXED_DEPTH_S
	case KPBoundaryCondition::FIXED_DEPTH:     BCKernelLauncher<N, BCFixedDepthSouth>(config, east, west); break;
#endif
#ifdef KPSIMULATOR_USE_BC_FIXED_DISCHARGE_S
	case KPBoundaryCondition::FIXED_DISCHARGE: BCKernelLauncher<N, BCFixedDischargeSouth>(config, east, west); break;
#endif
#ifdef KPSIMULATOR_USE_BC_OPEN_S
	case KPBoundaryCondition::OPEN:            BCKernelLauncher<N, BCOpenSouth>(config, east, west); break;
#endif
	default:
		std::stringstream log;
		log << __FILE__ << ":" << __LINE__ << " Boundary condition not compiled." << std::endl;
		throw std::runtime_error(log.str());
	}
}

/**
 * Used to translate runtime-parameter north to compile-time kernel for north boundary
 */
void BCKernelLauncher(const KernelConfiguration& config,
		KPBoundaryCondition::TYPE north,
		KPBoundaryCondition::TYPE south,
		KPBoundaryCondition::TYPE east,
		KPBoundaryCondition::TYPE west) {
	switch (north) {
	case KPBoundaryCondition::NONE:            BCKernelLauncher<BCNone>(config, south, east, west); break;
#ifdef KPSIMULATOR_USE_BC_WALL_N
	case KPBoundaryCondition::WALL:            BCKernelLauncher<BCWallNorth>(config, south, east, west); break;
#endif
#ifdef KPSIMULATOR_USE_BC_FIXED_DEPTH_N
	case KPBoundaryCondition::FIXED_DEPTH:     BCKernelLauncher<BCFixedDepthNorth>(config, south, east, west); break;
#endif
#ifdef KPSIMULATOR_USE_BC_FIXED_DISCHARGE_N
	case KPBoundaryCondition::FIXED_DISCHARGE: BCKernelLauncher<BCFixedDischargeNorth>(config, south, east, west); break;
#endif
#ifdef KPSIMULATOR_USE_BC_OPEN_N
	case KPBoundaryCondition::OPEN:            BCKernelLauncher<BCOpenNorth>(config, south, east, west); break;
#endif
	default:
		std::stringstream log;
		log << __FILE__ << ":" << __LINE__ << " Boundary condition not compiled." << std::endl;
		throw std::runtime_error(log.str());
	}
}







