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


#ifndef KPSIMULATOR_IMPL_H__
#define KPSIMULATOR_IMPL_H__

#include "KPTypes.h"

struct KPSimulator::Impl {
	static const unsigned int EULER_CTX = 0; //Offset into the param arrays to run an euler step
	static const unsigned int RK1_CTX   = 1; //Offset into the param arrays to run first rk substep
	static const unsigned int RK2_CTX   = 2; //Offset into the param arrays to run second rk substep

	Impl(const KPInitialConditions& ic_);
	~Impl();

	/**
	 * Initializing function that allocates data,
	 * and uploads initial conditions, etc
	 */
	void init();

	/**
	 * Initializes execution configurations for the kernels
	 */
	void init_kernel_params();

	/**
	 * Function that sets CUDA kernel arguments (pointers to data, etc)
	 */
	void init_kernel_args();

	/**
	 * Function that allocates all data we need, streams we need, etc
	 */
	void init_allocate();

	/**
	 * Function that uploads all data from initial conditions,
	 * and runs preprocessing before simulation can start (e.g. filling
	 * in ghost cells with proper values etc.)
	 */
	void init_data();

	/** 
	 * Initializes B at midpoints
	 */
	void init_Bm();

	/** 
	 * Initializes U.
	 */
	void init_U();
    
	/**
	 * Performs one timestep
	 */
	void step();

	/**
	 * Performs one timestep using first order Euler ode integration
	 */
	void eulerStep();

	/**
	 * Performs one timestep using 2nd order Runge-Kutta ode integration
	 */
	void rungeKutta2Step();
    
	void Dt(const DtLaunchConfig& params, const DtKernelArgs* args);
	void BC(const BCLaunchConfig& params, const BCKernelArgs* args);
	void RK1(const RKLaunchConfig& params, const RKKernelArgs* args, 
			const unsigned int domain_width=0, const unsigned int domain_height=0);
	void RK2(const RKLaunchConfig& params, const RKKernelArgs* args, 
			const unsigned int domain_width=0, const unsigned int domain_height=0);
	void FGH1(const FGHLaunchConfig& params, const FGHKernelArgs* args, 
			const unsigned int domain_width=0, const unsigned int domain_height=0);
	void FGH2(const FGHLaunchConfig& params, const FGHKernelArgs* args, 
			const unsigned int domain_width=0, const unsigned int domain_height=0);

	static void BCWallSetup(const shared_gpu_ptr_2D& Bi, const shared_gpu_ptr_2D& Bm, unsigned int width, unsigned int height);

	//Data allocated on the GPU
	boost::shared_ptr<gpu_ptr_2D<float> > U[3];   //!< U = [h, hu, hv]^n
	boost::shared_ptr<gpu_ptr_2D<float> > Q[3];   //!< Q = [h, hu, hv]^{n-0.5} (Runge kutta)
	boost::shared_ptr<gpu_ptr_2D<float> > R[3];   //!< R = Source term and net flux for each cell
	boost::shared_ptr<gpu_ptr_2D<float> > Bi;     //!< Bi = bottom topography at grid cell intersections
	boost::shared_ptr<gpu_ptr_2D<float> > Bm;     //!< Bm = bottom topography at grid cell midpoints
	boost::shared_ptr<gpu_ptr_2D<float> > M;      //!< M = Mannings N, the friction coefficient
	boost::shared_ptr<gpu_ptr_1D<float> > L;      //!< Eigenvalues per block used to calculate dt
	boost::shared_ptr<gpu_ptr_2D<float> > D;      //!< dry map per block.
	boost::shared_ptr<gpu_ptr_1D<float> > dt_d;   //!< device pointer to delta t
	
	BCLaunchConfig bc_params;           //!< Parameters to the boundary conditions kernel
	FGHLaunchConfig flux_params;        //!< Parameters to the flux kernel
	RKLaunchConfig rk_params;           //!< Parameters to the runge-kutta kernel
	DtLaunchConfig dt_params;           //!< Parameters to the delta t kernel

	FGHLaunchConfig mod_flux_params;	//!< Modified kernel execution configuration for flux kernel (just enough blocks).
	RKLaunchConfig mod_rk_params;       //!< Modified kernel execution configuration for rk kernel (just enough blocks).

	BCKernelArgs* bc_args[3];            //!< Arguments to the boundary conditions kernel
	FGHKernelArgs* flux_args[3];         //!< Arguments to the flux kernel
	RKKernelArgs* rk_args[3];            //!< Arguments to the runge-kutta kernel
	DtKernelArgs* dt_args;               //!< Arguments to the delta t kernel
	
	KPInitialConditions ic;             //!< Initial conditions
	long step_counter;                  //!< The current simulation step counter @see getTimeSteps()
	double time;                        //!< The current simulation time @see getTime()
	float* dt_h;                        //!< Page locked memory used to transfer dt from device to host.

	cudaStream_t streams[2];            //!< Asynchronous streams for GPU execution.

	float autotune_step_time;           //!< Cumulative time spent in the step function by the GPU 
	cudaEvent_t step_start, step_stop;  //!< timers to time the execution of the flux kernel
	cudaEvent_t stream_sync;            //!< Synchronization between streams
	cudaEvent_t dt_complete;            //!< Event to make sure dt has been computed.
};

#endif