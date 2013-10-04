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
#include "configure.h"
#include "KPSimulatorHelpers.hpp"
#include "KPException.hpp"
#include "cuda/util.h"

#include "cuda/kernel/Dt.hpp"
#include "cuda/kernel/RK.hpp"
#include "cuda/kernel/FGH.hpp"
#include "cuda/kernel/InitBm.hpp"
#include "cuda/kernel/InitU.hpp"

#include <cmath>
#include <omp.h>
#include <time.h>
#include <fstream>

#ifndef NDEBUG
#include <iomanip>
#include <iostream>
#endif

#define USE_1D_INDEX
#define STRINGIFY(str) STRINGIFY_HELPER(str)
#define STRINGIFY_HELPER(str) #str

	
KPSimulator::Impl::Impl(const KPInitialConditions& ic_) : ic(ic_), 
			step_counter(0), time(0.0f), 
			autotune_step_time(0.0f) {
	init();
}

KPSimulator::Impl::~Impl() {
	for (int i=0; i<3; ++i) {
		KPSIMULATOR_CHECK_CUDA(cudaFreeHost(bc_args[i]));
		KPSIMULATOR_CHECK_CUDA(cudaFreeHost(flux_args[i]));
		KPSIMULATOR_CHECK_CUDA(cudaFreeHost(rk_args[i]));
	}
	KPSIMULATOR_CHECK_CUDA(cudaFreeHost(dt_args));
	KPSIMULATOR_CHECK_CUDA(cudaFreeHost(dt_h));

	for (int i=0; i<2; ++i)
		KPSIMULATOR_CHECK_CUDA(cudaStreamDestroy(streams[i]));
	
	KPSIMULATOR_CHECK_CUDA(cudaEventDestroy(step_start));
	KPSIMULATOR_CHECK_CUDA(cudaEventDestroy(step_stop));
	KPSIMULATOR_CHECK_CUDA(cudaEventDestroy(stream_sync));
	KPSIMULATOR_CHECK_CUDA(cudaEventDestroy(dt_complete));

	for (int i=0; i<3; ++i) {
		U[i].reset();
		Q[i].reset();
		R[i].reset();
	}
	
	Bi.reset();
	Bm.reset();
	D.reset();
	L.reset();
	dt_d.reset();
}




using boost::shared_ptr;

void BCKernelLauncher(const BCKernelArgs* h_ctx,
		const KernelConfiguration& config,
		const cudaStream_t& stream,
		KPBoundaryCondition::TYPE north,
		KPBoundaryCondition::TYPE south,
		KPBoundaryCondition::TYPE east,
		KPBoundaryCondition::TYPE west);

KPSimulator::KPSimulator(const KPInitialConditions& ic_) {
	pimpl.reset(new Impl(ic_));
}

KPSimulator::~KPSimulator() {
	pimpl.reset();
}

void KPSimulator::Impl::eulerStep() {
    FGH1(flux_params, flux_args[EULER_CTX]);
    Dt(dt_params, dt_args);
	cudaEventRecord(dt_complete, streams[0]);
    RK1(rk_params, rk_args[EULER_CTX]);
	cudaEventRecord(stream_sync, streams[0]);

	KPSIMULATOR_CHECK_CUDA(cudaStreamWaitEvent(streams[0], stream_sync, 0)); //FIXME: see sources
	
	BC(bc_params, bc_args[EULER_CTX]);
	
	KPSIMULATOR_CHECK_CUDA(cudaEventSynchronize(dt_complete));
    time += *dt_h;
}

void KPSimulator::Impl::rungeKutta2Step() {
	FGH1(flux_params, flux_args[RK1_CTX]);
	Dt(dt_params, dt_args);
	cudaEventRecord(dt_complete, streams[0]);
	RK1(rk_params, rk_args[RK1_CTX]);
		
	BC(bc_params, bc_args[RK1_CTX]);
		
	FGH2(flux_params, flux_args[RK2_CTX]);
	RK2(rk_params, rk_args[RK2_CTX]);
	cudaEventRecord(stream_sync, streams[0]);
	
	KPSIMULATOR_CHECK_CUDA(cudaStreamWaitEvent(streams[0], stream_sync, 0)); //FIXME: see sources
	
	BC(bc_params, bc_args[RK2_CTX]);
	
	KPSIMULATOR_CHECK_CUDA(cudaEventSynchronize(dt_complete));
	time += *dt_h;
}


void KPSimulator::Impl::step() {
	float step_time;

	//Update boundary condition values
	for (unsigned int k=0; k<3; ++k) {
		bc_args[k]->north_arg = ic.getNorthBC().getValue(static_cast<float>(time));
		bc_args[k]->south_arg = ic.getSouthBC().getValue(static_cast<float>(time));
		bc_args[k]->east_arg = ic.getEastBC().getValue(static_cast<float>(time));
		bc_args[k]->west_arg = ic.getWestBC().getValue(static_cast<float>(time));
	}

	cudaEventRecord(step_start, streams[0]);

	switch (ic.getTimeIntegrator()) {
	case KPInitialConditions::EULER: eulerStep(); break;
	case KPInitialConditions::RUNGE_KUTTA_2: rungeKutta2Step(); break;
	default: std::cout << "Unknown time integrator" << std::endl; exit(-1);
	}

	cudaEventRecord(step_stop, streams[0]);
	cudaEventSynchronize(step_stop);
	cudaEventElapsedTime(&step_time, step_start, step_stop);
	autotune_step_time += step_time;
	
	++step_counter;
}

void KPSimulator::getB(float* data,
		unsigned int nx_offset, unsigned int ny_offset,
		unsigned int nx, unsigned int ny) {
	//Downloads at midpoints
	nx = (nx > 0) ? nx : pimpl->ic.getUNx();
	ny = (ny > 0) ? ny : pimpl->ic.getUNy();
	pimpl->Bm->download(data, nx_offset, ny_offset, nx, ny);
}

void KPSimulator::getU1(float* data,
		unsigned int nx_offset, unsigned int ny_offset,
		unsigned int nx, unsigned int ny) {
	nx = (nx > 0) ? nx : pimpl->ic.getUNx();
	ny = (ny > 0) ? ny : pimpl->ic.getUNy();
	pimpl->U[0]->download(data, nx_offset, ny_offset, nx, ny);
}

void KPSimulator::getU2(float* data,
		unsigned int nx_offset, unsigned int ny_offset,
		unsigned int nx, unsigned int ny) {
	nx = (nx > 0) ? nx : pimpl->ic.getUNx();
	ny = (ny > 0) ? ny : pimpl->ic.getUNy();
	pimpl->U[1]->download(data, nx_offset, ny_offset, nx, ny);
}

void KPSimulator::getU3(float* data,
		unsigned int nx_offset, unsigned int ny_offset,
		unsigned int nx, unsigned int ny) {
	nx = (nx > 0) ? nx : pimpl->ic.getUNx();
	ny = (ny > 0) ? ny : pimpl->ic.getUNy();
	pimpl->U[2]->download(data, nx_offset, ny_offset, nx, ny);
}

void KPSimulator::setU1(const float* data,
		unsigned int nx_offset, unsigned int ny_offset,
		unsigned int nx, unsigned int ny) {
	nx = (nx > 0) ? nx : pimpl->ic.getUNx();
	ny = (ny > 0) ? ny : pimpl->ic.getUNy();
	pimpl->U[0]->upload(data, nx_offset, ny_offset, nx, ny);
	pimpl->init_U();

	if (pimpl->ic.getTimeIntegrator() == KPInitialConditions::RUNGE_KUTTA_2) {
		for (unsigned int i=0; i<3; ++i) //<FIXME: Can we remove this?
			pimpl->Q[i]->copy(*pimpl->U[i].get());
	}
}

void KPSimulator::setU2(const float* cpu_data,
			unsigned int nx_offset, unsigned int ny_offset,
			unsigned int nx, unsigned int ny) {
	nx = (nx > 0) ? nx : pimpl->ic.getUNx();
	ny = (ny > 0) ? ny : pimpl->ic.getUNy();
	pimpl->U[1]->upload(cpu_data, nx_offset, ny_offset, nx, ny);
}

void KPSimulator::setU3(const float* cpu_data,
			unsigned int nx_offset, unsigned int ny_offset,
			unsigned int nx, unsigned int ny) {
	nx = (nx > 0) ? nx : pimpl->ic.getUNx();
	ny = (ny > 0) ? ny : pimpl->ic.getUNy();
	pimpl->U[2]->upload(cpu_data, nx_offset, ny_offset, nx, ny);
}

void KPSimulator::setB(const float* data,
		unsigned int nx_offset, unsigned int ny_offset,
		unsigned int nx, unsigned int ny) {
	nx = (nx > 0) ? nx : pimpl->ic.getBNx();
	ny = (ny > 0) ? ny : pimpl->ic.getBNy();
	pimpl->Bi->upload(data, nx_offset, ny_offset, nx, ny);
	pimpl->init_Bm(); //FIXME: This updates all of Bm, while we can get away with just updating nx by ny cells...
}





void KPSimulator::Impl::init() {
	//Set up and allocate cuda streams
	for (int i=0; i<2; ++i)
		KPSIMULATOR_CHECK_CUDA(cudaStreamCreate(&streams[i]));
	
	KPSIMULATOR_CHECK_CUDA(cudaEventCreate(&step_start));
	KPSIMULATOR_CHECK_CUDA(cudaEventCreate(&step_stop));
	KPSIMULATOR_CHECK_CUDA(cudaEventCreate(&stream_sync));
	KPSIMULATOR_CHECK_CUDA(cudaEventCreate(&dt_complete));

	init_kernel_params();
	init_kernel_args();
}




void KPSimulator::Impl::init_kernel_params() {
	/**
	  * Set block configuration for the different kernels
	  * This also forms the basis for calculating how large our padded
	  * domain is
	  */
	rk_params.config = get_rk_config(ic.getNx(), ic.getNy(), streams[0]);

	bc_params.config = get_bc_config(std::max(ic.getNx(), ic.getNy()), streams[0]);
	bc_params.north = ic.getNorthBC().getType();
	bc_params.south= ic.getSouthBC().getType();
	bc_params.east = ic.getEastBC().getType();
	bc_params.west = ic.getWestBC().getType();

	flux_params.config = get_flux_config(ic.getNx(), ic.getNy(), streams[0]);
    
	dt_params.config = get_dt_config(flux_params.config.grid.x*flux_params.config.grid.y, streams[0]);
}






void KPSimulator::Impl::init_allocate() {
	/**
	 * FIXME: Optimize this
	 * Also include ghost cells in the domain width calculation...
	 * We have nx+1 intersections, and 4 ghost values for our largest domain,
	 * B given at intersections.
	 */
	dim3 domain;
	domain.x = std::max(rk_params.config.block.x*rk_params.config.grid.x, flux_params.config.block.x*flux_params.config.grid.x) + 5;
	domain.y = std::max(rk_params.config.block.y*rk_params.config.grid.y, flux_params.config.block.y*flux_params.config.grid.y) + 5;

	// First, allocate all the data that we need on the gpu
	for (int i=0; i<3; ++i) {
		U[i].reset(new gpu_ptr_2D<float>(domain.x, domain.y));
		Q[i].reset(new gpu_ptr_2D<float>(domain.x, domain.y));
		R[i].reset(new gpu_ptr_2D<float>(domain.x, domain.y));
	}	

	if (ic.getM().spatially_varying) {
		M.reset(new gpu_ptr_2D<float>(domain.x, domain.y));
	}
	else {
		M.reset(new gpu_ptr_2D<float>(1, 1));
	}
	Bi.reset(new gpu_ptr_2D<float>(domain.x, domain.y));
	Bm.reset(new gpu_ptr_2D<float>(domain.x, domain.y));
	D.reset(new gpu_ptr_2D<float>(flux_params.config.grid.x, flux_params.config.grid.y));
	L.reset(new gpu_ptr_1D<float>(flux_params.config.grid.x*flux_params.config.grid.y));
	dt_d.reset(new gpu_ptr_1D<float>(1));
	    	
	//Allocate host dt
#if CUDA_40
	KPSIMULATOR_CHECK_CUDA(cudaHostAlloc(&dt_h, sizeof(float), cudaHostAllocDefault));

	//Allocate host data for kernel arguments
	for (int i=0; i<3; ++i) { 
		KPSIMULATOR_CHECK_CUDA(cudaHostAlloc(&bc_args[i], sizeof(BCKernelArgs), cudaHostAllocWriteCombined));
		KPSIMULATOR_CHECK_CUDA(cudaHostAlloc(&flux_args[i], sizeof(FGHKernelArgs), cudaHostAllocWriteCombined));
		KPSIMULATOR_CHECK_CUDA(cudaHostAlloc(&rk_args[i], sizeof(RKKernelArgs), cudaHostAllocWriteCombined));
		KPSIMULATOR_CHECK_CUDA(cudaHostAlloc(&source_args[i], sizeof(SourceKernelArgs), cudaHostAllocWriteCombined));
	}
	KPSIMULATOR_CHECK_CUDA(cudaHostAlloc(&dt_args, sizeof(DtKernelArgs), cudaHostAllocWriteCombined));
#else
	KPSIMULATOR_CHECK_CUDA(cudaMallocHost(&dt_h, sizeof(float)));

	//Allocate host data for kernel arguments
	for (int i=0; i<3; ++i) { 
		KPSIMULATOR_CHECK_CUDA(cudaMallocHost(&bc_args[i], sizeof(BCKernelArgs)));
		KPSIMULATOR_CHECK_CUDA(cudaMallocHost(&flux_args[i], sizeof(FGHKernelArgs)));
		KPSIMULATOR_CHECK_CUDA(cudaMallocHost(&rk_args[i], sizeof(RKKernelArgs)));
	}
	KPSIMULATOR_CHECK_CUDA(cudaMallocHost(&dt_args, sizeof(DtKernelArgs)));
#endif
}






void KPSimulator::Impl::init_kernel_args() {
	//First, allocate GPU data
	init_allocate();

	//Set parameters for the flux kernel
	set_flux_args(flux_args[EULER_CTX],
			U[0]->getRawPtr(), U[1]->getRawPtr(), U[2]->getRawPtr(),
			R[0]->getRawPtr(), R[1]->getRawPtr(), R[2]->getRawPtr(),
			D->getRawPtr(),
			Bi->getRawPtr(),
			Bm->getRawPtr(),
			L->getRawPtr(),
			ic.getDx(), ic.getDy(),
			ic.getNx(), ic.getNy(),
			ic.getG());
	set_flux_args(flux_args[RK1_CTX],
			U[0]->getRawPtr(), U[1]->getRawPtr(), U[2]->getRawPtr(),
			R[0]->getRawPtr(), R[1]->getRawPtr(), R[2]->getRawPtr(),
			D->getRawPtr(),
			Bi->getRawPtr(),
			Bm->getRawPtr(),
			L->getRawPtr(),
			ic.getDx(), ic.getDy(),
			ic.getNx(), ic.getNy(),
			ic.getG());
	set_flux_args(flux_args[RK2_CTX],
			Q[0]->getRawPtr(), Q[1]->getRawPtr(), Q[2]->getRawPtr(),
			R[0]->getRawPtr(), R[1]->getRawPtr(), R[2]->getRawPtr(),
			D->getRawPtr(),
			Bi->getRawPtr(),
			Bm->getRawPtr(),
			L->getRawPtr(),
			ic.getDx(), ic.getDy(),
			ic.getNx(), ic.getNy(),
			ic.getG());

	//Set parameters for runge-kutta kernel
	set_rk_args(rk_args[EULER_CTX],
			U[0]->getRawPtr(), U[1]->getRawPtr(), U[2]->getRawPtr(),
			U[0]->getRawPtr(), U[1]->getRawPtr(), U[2]->getRawPtr(),
			R[0]->getRawPtr(), R[1]->getRawPtr(), R[2]->getRawPtr(),
			M->getRawPtr(),
			D->getRawPtr(),
			Bm->getRawPtr(),
			dt_d->getRawPtr(),
			ic.getNx(), ic.getNy(),
			ic.getG(),
			ic.getM().spatially_varying);
	set_rk_args(rk_args[RK1_CTX],
			U[0]->getRawPtr(), U[1]->getRawPtr(), U[2]->getRawPtr(),
			Q[0]->getRawPtr(), Q[1]->getRawPtr(), Q[2]->getRawPtr(),
			R[0]->getRawPtr(), R[1]->getRawPtr(), R[2]->getRawPtr(),
			M->getRawPtr(),
			D->getRawPtr(),
			Bm->getRawPtr(),
			dt_d->getRawPtr(),
			ic.getNx(), ic.getNy(),
			ic.getG(),
			ic.getM().spatially_varying);
	set_rk_args(rk_args[RK2_CTX],
			Q[0]->getRawPtr(), Q[1]->getRawPtr(), Q[2]->getRawPtr(),
			U[0]->getRawPtr(), U[1]->getRawPtr(), U[2]->getRawPtr(),
			R[0]->getRawPtr(), R[1]->getRawPtr(), R[2]->getRawPtr(),
			M->getRawPtr(),
			D->getRawPtr(),
			Bm->getRawPtr(),
			dt_d->getRawPtr(),
			ic.getNx(), ic.getNy(),
			ic.getG(),
			ic.getM().spatially_varying);

	//Set parameters for boundary conditions
	set_bc_args(bc_args[EULER_CTX],
			U[0]->getRawPtr(), U[1]->getRawPtr(), U[2]->getRawPtr(),
			ic.getNx(), ic.getNy());
	set_bc_args(bc_args[RK1_CTX],
			Q[0]->getRawPtr(), Q[1]->getRawPtr(), Q[2]->getRawPtr(),
			ic.getNx(), ic.getNy());
	set_bc_args(bc_args[RK2_CTX],
			U[0]->getRawPtr(), U[1]->getRawPtr(), U[2]->getRawPtr(),
			ic.getNx(), ic.getNy());

	//Set parameters for dt-kernel
	set_dt_args(dt_args, L->getRawPtr(), L->getWidth(), dt_d->getRawPtr(),
			ic.getDx(), ic.getDy(), ic.getDtScale());

	//Finally, set the data
	init_data();
}


void KPSimulator::Impl::init_data() {
	//Initialize GPU data
	for (int i=0; i<3; ++i)
		R[i]->set(0);
	L->set(FLT_MAX);
	D->set(0.0f); //< Now used in grow-kernel
	
	dt_d->set(FLT_MAX);
	*dt_h = 0.0f;

	//Upload U and B
	for (unsigned int i=0; i<3; ++i)
		U[i]->set(0);
	for (unsigned int i=0; i<3; ++i)
		Q[i]->set(0);
	U[0]->upload(ic.getU1(), 2, 2, ic.getUNx(), ic.getUNy());
	if (ic.getU2() != NULL) U[1]->upload(ic.getU2(), 2, 2, ic.getUNx(), ic.getUNy());
	if (ic.getU3() != NULL) U[2]->upload(ic.getU3(), 2, 2, ic.getUNx(), ic.getUNy());
	Bi->upload(ic.getB(), 2, 2, ic.getBNx(), ic.getBNy());

	//Upload M
	if (ic.getM().spatially_varying) {
		M->upload(ic.getM().n, 2, 2, ic.getUNx(), ic.getUNy());
	}
	else {
		M->upload(ic.getM().n, 0, 0, 1, 1);
	}

	//FIXME: Global boundary not initialized proper when using KPBoundaryCondition::NONE
    
	//Set boundary conditions to fill inn ghost cells
	BCWallSetup(Bi, Bm, ic.getNx(), ic.getNy());

	//Initialize Bm and U
	init_Bm();
	init_U();

    //Initialize boundaries
	BC(bc_params, bc_args[EULER_CTX]);
	BC(bc_params, bc_args[RK1_CTX]);
}




void KPSimulator::Impl::Dt(const DtLaunchConfig& params, const DtKernelArgs* args) {
	//Copy parameters to the GPU
	KPSIMULATOR_CHECK_CUDA(cudaMemcpyToSymbolAsync(dt_ctx, args, sizeof(DtKernelArgs), 0, cudaMemcpyHostToDevice, params.config.stream));

	//Launch kernel
	switch(params.config.block.x) {
		case 512: dtKernelLauncher<512>(params.config); break;
		case 256: dtKernelLauncher<256>(params.config); break;
		case 128: dtKernelLauncher<128>(params.config); break;
		case  64: dtKernelLauncher<64>(params.config); break;
		case  32: dtKernelLauncher<32>(params.config); break;
		case  16: dtKernelLauncher<16>(params.config); break;
		case   8: dtKernelLauncher<8>(params.config); break;
		case   4: dtKernelLauncher<4>(params.config); break;
		case   2: dtKernelLauncher<2>(params.config); break;
		case   1: dtKernelLauncher<1>(params.config); break;
	}

	//Copy dt back to the cpu
	KPSIMULATOR_CHECK_CUDA(cudaMemcpyAsync(dt_h, dt_d->getRawPtr(), sizeof(float), cudaMemcpyDeviceToHost, dt_params.config.stream));
}

void KPSimulator::Impl::RK1(const RKLaunchConfig& params, const RKKernelArgs* args, 
		const unsigned int domain_width, const unsigned int domain_height) {
	RKKernelLauncher<0>(args, params.config); 
}

void KPSimulator::Impl::RK2(const RKLaunchConfig& params, const RKKernelArgs* args, 
		const unsigned int domain_width, const unsigned int domain_height) {
	RKKernelLauncher<1>(args, params.config); 
}

void KPSimulator::Impl::FGH1(const FGHLaunchConfig& params, const FGHKernelArgs* args, 
		const unsigned int domain_width, const unsigned int domain_height) {
	FGHKernelLauncher<0>(args, params.config); 
}

void KPSimulator::Impl::FGH2(const FGHLaunchConfig& params, const FGHKernelArgs* args, 
		const unsigned int domain_width, const unsigned int domain_height) {
	FGHKernelLauncher<1>(args, params.config); 
}

void KPSimulator::Impl::init_Bm() { //Bm, Bi
	const unsigned int block_width = 16;
	const unsigned int block_height = 16;
	unsigned int width = Bi->getWidth();
	unsigned int height = Bi->getHeight();
	dim3 block = dim3(block_width, block_height);
	dim3 grid = dim3((width + (block.x - 1)) / block.x, (height + (block.y - 1)) / block.y);

	initBmKernel<block_width, block_height><<<grid, block>>>(Bm->getRawPtr(), Bi->getRawPtr(), width, height);
	KPSIMULATOR_CHECK_CUDA_ERROR("initBmKernel");
}


void KPSimulator::Impl::init_U() { //U, Bm
	const unsigned int block_width = 16;
	const unsigned int block_height = 16;
	dim3 block, grid;
	unsigned int width;
	unsigned int height;
	gpu_raw_ptr<> U1_ptr, U2_ptr, U3_ptr, Bm_ptr;

	width = U[0]->getWidth();
	height = U[0]->getHeight();

	U1_ptr = U[0]->getRawPtr();
	U2_ptr = U[1]->getRawPtr();
	U3_ptr = U[2]->getRawPtr();
	Bm_ptr = Bm->getRawPtr();

	block = dim3(block_width, block_height);
	grid = dim3((width + (block.x - 1)) / block.x, (height + (block.y - 1)) / block.y);

	initUKernel<block_width, block_height><<<grid, block>>>(U1_ptr, U2_ptr, U3_ptr, Bm_ptr, width, height);
	KPSIMULATOR_CHECK_CUDA_ERROR("initUKernel");
}



const char* KPSimulator::versionString() {
	return STRINGIFY(KPSIMULATOR_MAJOR_VERSION) 
			"." STRINGIFY(KPSIMULATOR_MINOR_VERSION) 
			"." STRINGIFY(KPSIMULATOR_BUILD_VERSION); 
}

unsigned int KPSimulator::majorVersion() {
	return KPSIMULATOR_MAJOR_VERSION;
}

unsigned int KPSimulator::minorVersion() {
	return KPSIMULATOR_MINOR_VERSION;
}

unsigned int KPSimulator::buildVersion() {
	return KPSIMULATOR_BUILD_VERSION;
}


std::ostream& KPSimulator::print(std::ostream& out) const {
	out << "=== KPSimulator (" << KPSimulator::versionString() << ") ===" << std::endl;
	out << "Compiled " << __TIME__ " " << __DATE__ << "." << std::endl;
	out << "Flux block size = [" << KPSIMULATOR_FLUX_BLOCK_WIDTH << "x" << KPSIMULATOR_FLUX_BLOCK_HEIGHT << "]" << std::endl;
	out << "RK block size   = [" << KPSIMULATOR_RK_BLOCK_WIDTH << "x" << KPSIMULATOR_RK_BLOCK_HEIGHT << "]" << std::endl;
	out << "Minmod theta    = " << KPSIMULATOR_MINMOD_THETA << std::endl;
	out << "Flux slope eps  = " << KPSIMULATOR_FLUX_SLOPE_EPS << std::endl;
	out << "Zero flux eps   = " << KPSIMULATOR_ZERO_FLUX_EPS << std::endl;
	out << "Dry eps         = " << KPSIMULATOR_DRY_EPS << std::endl;
	out << "CXX compiler    = " << KPSIMULATOR_CXX_COMPILER_STRING << std::endl;
	out << "CUDA compiler   = " << KPSIMULATOR_CUDA_COMPILER_STRING << std::endl;
	out << std::endl;
	out << "=== Initial Conditions ===" << std::endl;
	out << pimpl->ic << std::endl;
	return out;
}

std::ostream& operator<<(std::ostream& out, const KPSimulator& sim) {
	sim.print(out);
	return out;
}

std::ostream& operator<<(std::ostream& out, const KPSimulator* sim) {
	sim->print(out);
	return out;
}











long KPSimulator::getTimeSteps() const {
	return pimpl->step_counter; 
}

float KPSimulator::getTime() const {
	return (float) pimpl->time;
}

float KPSimulator::getDt() const { 
	return (float) *pimpl->dt_h;
}

const KPInitialConditions& KPSimulator::getIC() const {
	return pimpl->ic; 
}

boost::shared_ptr<gpu_ptr_2D<float> > KPSimulator::getBm() {
	return pimpl->Bm;
}

boost::shared_ptr<gpu_ptr_2D<float> > KPSimulator::getR1() {
	return pimpl->R[0];
}

boost::shared_ptr<gpu_ptr_2D<float> > KPSimulator::getR2() {
	return pimpl->R[1]; 
}

boost::shared_ptr<gpu_ptr_2D<float> > KPSimulator::getR3() {
	return pimpl->R[2]; 
}

boost::shared_ptr<gpu_ptr_2D<float> > KPSimulator::getU1() { 
	return pimpl->U[0]; 
}

boost::shared_ptr<gpu_ptr_2D<float> > KPSimulator::getU2() { 
	return pimpl->U[1]; 
}

boost::shared_ptr<gpu_ptr_2D<float> > KPSimulator::getU3() { 
	return pimpl->U[2];
}

boost::shared_ptr<gpu_ptr_2D<float> > KPSimulator::getD() { 
	return pimpl->D;
}

boost::shared_ptr<gpu_ptr_1D<float> > KPSimulator::getL() {
	return pimpl->L; 
}

void KPSimulator::step() {
	pimpl->step();
}
