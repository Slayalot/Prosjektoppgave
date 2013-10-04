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

#include "KPSimulator.h"
#include "KPSimulatorImpl.h"
#include "KPException.hpp"
#include "cuda/kernel/BC.hpp"

/**
 * FIXME: Make this work for all types of boundaries, not only wall
 * or even better... integrate into initBm/initBi
 */
void KPSimulator::Impl::BCWallSetup(const shared_gpu_ptr_2D& Bi, const shared_gpu_ptr_2D& Bm, unsigned int width, unsigned int height) {
	dim3 block, grid;

	grid = dim3(1, 1, 1);
#if __CUDA_ARCH__ >= 200
	block = dim3(512, 1, 1);
#else
	block = dim3(256, 1, 1);
#endif
	switch(block.x) {
		case 512: BCWallSetupKernel<512><<<grid, block>>>(Bi->getRawPtr(), Bm->getRawPtr(), width, height); break;
		case 256: BCWallSetupKernel<256><<<grid, block>>>(Bi->getRawPtr(), Bm->getRawPtr(), width, height); break;
	}
	KPSIMULATOR_CHECK_CUDA_ERROR("BCWallSetupKernel");
}

void KPSimulator::Impl::BC(const BCLaunchConfig& params, const BCKernelArgs* args) {
	//Skip launching a kernel if we don not care about boundary conditions
	if (params.north == KPBoundaryCondition::NONE &&
			params.south == KPBoundaryCondition::NONE &&
			params.east == KPBoundaryCondition::NONE &&
			params.west == KPBoundaryCondition::NONE) return;
	
	//Copy parameters to the GPU
	KPSIMULATOR_CHECK_CUDA(cudaMemcpyToSymbolAsync(bc_ctx, args, sizeof(BCKernelArgs), 0, cudaMemcpyHostToDevice, params.config.stream));
	BCKernelLauncher(params.config, params.north, params.south, params.east, params.west);
}

