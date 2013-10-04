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

#ifndef KPTYPES_H_
#define KPTYPES_H_

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
#include <limits>
#include <iostream>
#include <vector>
#include <boost/utility.hpp>
#include <cassert>
#include <cuda_runtime.h>

#include "gpu_raw_ptr.hpp"
#include "KPBoundaryConditions.h"

/**
 * Struct that keeps track of block and grid sizes for different
 * kernel launches
 */
struct KernelConfiguration {
	dim3 block;
	dim3 grid;
	cudaStream_t stream;
};

/**
 * Helper function to easily print out kernel configurations
 */
inline std::ostream& operator<<(std::ostream& out, const KernelConfiguration& config) {
	out << "Kernel config: ";
	out << "<<<[" << config.grid.x << "x" << config.grid.y << "x" << config.grid.z << "], ";
	out << "[" << config.block.x << "x" << config.block.y << "x" << config.block.z << "], ";
	out << config.stream << ">>>";
#ifndef NDEBUG
	out << " (" << config.block.x*config.grid.x << "x" << config.block.y*config.grid.y << "x" << config.block.z*config.grid.z << ")";
#endif
	return out;
}

/**
 * Parameters used by the boundary-conditions kernel
 */
struct BCKernelArgs {
	gpu_raw_ptr<> U1, U2, U3; //!< input/output
	float north_arg, south_arg, east_arg, west_arg;
	unsigned int width, height;
};

struct BCLaunchConfig {
	KernelConfiguration config;
	KPBoundaryCondition::TYPE north, south, east, west;
};

/**
 * Parameters used by the maximum dt-kernel
 */
struct DtKernelArgs {
	float* L;               //!< Maximal eigenvalues for each block
	float* dt;              //!< Output delta t
	unsigned int elements;	//!< Elements in eigenvalue buffer
	float dx;         		//!< Spatial distance between cells in x direction
	float dy;         		//!< Spatial distance between cells in y direction
	float scale;            //!< Scaling of dt to guarantee to maintain stable solution
};

struct DtLaunchConfig {
	KernelConfiguration config;
};

/**
 * Parameters used by the runge-kutta kernel
 */
struct RKKernelArgs {
	gpu_raw_ptr<>  Q1, Q2, Q3; //!< Newly calculated Q-vector (and input Q-vector for second step of RK)
	gpu_raw_ptr<>  U1, U2, U3; //!< U-vector at current timestep
	gpu_raw_ptr<>  R1, R2, R3; //!< Net flux in and out of cells
	gpu_raw_ptr<>  Bm;   //!< B at midpoints
	gpu_raw_ptr<> D;     //!< Dry map per block
	float *active_compact_x, *active_compact_y;//!< map of active blocks for current timestep
	gpu_raw_ptr<> M;     //!< Mannings N (friction coefficient)
	float* dt;         //!< Timestep
	float g;           //!< Gravitational constant
	bool spatially_varying_manning; //!< Is the manning coefficient constant or varying?
	unsigned int nx;   //!< Computational domain widht
	unsigned int ny;   //!< Computational domain height
};

struct RKLaunchConfig {
	KernelConfiguration config;
};

/**
 * Parameters used by the flux kernel
 */
struct FGHKernelArgs {
	gpu_raw_ptr<> U1, U2, U3;        //!< U vector given at cell midpoints
	gpu_raw_ptr<> R1, R2, R3;        //!< Source term and net flux in and out of each cell.
	gpu_raw_ptr<> Bi;                //!< Bathymetry given at cell intersections
	gpu_raw_ptr<> Bm;                //!< B at midpoints
	gpu_raw_ptr<> D;                 //!< Dry map used for early exit test
	float *active_compact_x, *active_compact_y;      //!< compacted map of active blocks for current timestep
	float* L;                      //!< Maximal eigenvalues for each block
	float dx;                      //!< Spatial distance between cells in x direction
	float dy;                      //!< Spatial distance between cells in y direction
	unsigned int nx;               //!< Domain size without ghost cells
	unsigned int ny;               //!< Domain size without ghost cells
	float g;                       //!< Gravitational constant
};

struct FGHLaunchConfig {
	KernelConfiguration config;
};




#endif /* KPTYPES_H_ */
