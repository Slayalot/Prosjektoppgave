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

#ifndef KPSIMULATOR_H_
#define KPSIMULATOR_H_

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#endif

#include <limits>
#include <boost/utility.hpp>
#include <boost/shared_ptr.hpp>
#include "KPInitialConditions.h"
#include "gpu_ptr.hpp"

/**
 * This class is the heart of our simulation. It defines the KPGPUSimulator
 * (Kurganov-Petrova GPU simulator), which exposes all functionality
 * required to run a GPU simulation.
 */
class KPSimulator : public boost::noncopyable {
public:
	/**
	 * Constructor that allocates data and initializes the domain.
	 */
	KPSimulator(const KPInitialConditions& init_);

	/**
	 * Destructor that frees up GPU data
	 */
	virtual ~KPSimulator();

	/**
	 * Performs one full simulation timestep
	 */
	void step();

	/**
	 * Returns the number of full timesteps in current
	 * simulation
	 * @return number of timesteps
	 */
	long getTimeSteps() const;

	/**
	 * Returns the time the simulation is at now (not wall time, but simulation time, as in
	 * average delta t times number of timesteps)
	 * @return simulation time
	 */
	float getTime() const;

	/**
	 * Returns the last Dt used in time integration
	 * @return timestep
	 */
	float getDt() const;

	/**
	 * Returns the initial conditions used in this simulator.
	 * @return Initial conditions handed to constructor
	 */
	const KPInitialConditions& getIC() const;

	/**
	 * Returns the gpu pointer to the height map data on the GPU
	 * @see gpu_ptr_2D
	 * @return pointer to the GPU memory (NOT accessible by the CPU directly)
	 */
	boost::shared_ptr<gpu_ptr_2D<float> > getBm();

	/**
	 * Returns the gpu pointer to the computed fluxes on the GPU
	 * @see gpu_ptr_2D
	 * @return pointer to the GPU memory (NOT accessible by the CPU directly)
	 */
	boost::shared_ptr<gpu_ptr_2D<float> > getR1();

	/**
	 * Returns the gpu pointer to the computed fluxes on the GPU
	 * @see gpu_ptr_2D
	 * @return pointer to the GPU memory (NOT accessible by the CPU directly)
	 */
	boost::shared_ptr<gpu_ptr_2D<float> > getR2();

	/**
	 * Returns the gpu pointer to the computed fluxes on the GPU
	 * @see gpu_ptr_2D
	 * @return pointer to the GPU memory (NOT accessible by the CPU directly)
	 */
	boost::shared_ptr<gpu_ptr_2D<float> > getR3();

	/**
	 * Returns the gpu pointer to the water elevation data on the GPU
	 * @see gpu_ptr_2D
	 * @return pointer to the GPU memory (NOT accessible by the CPU directly)
	 */
	boost::shared_ptr<gpu_ptr_2D<float> > getU1();

	/**
	 * Returns the gpu pointer to the U-discharge data on the GPU
	 * @see gpu_ptr_2D
	 * @return pointer to the GPU memory (NOT accessible by the CPU directly)
	 */
	boost::shared_ptr<gpu_ptr_2D<float> > getU2();

	/**
	 * Returns the gpu pointer to the V-discharge data on the GPU
	 * @see gpu_ptr_2D
	 * @return pointer to the GPU memory (NOT accessible by the CPU directly)
	 */
	boost::shared_ptr<gpu_ptr_2D<float> > getU3();

	/**
	 * Returns the gpu pointer to the dry map per block on the GPU
	 * @see gpu_ptr_2D
	 * @return pointer to the GPU memory (NOT accessible by the CPU directly)
	 */
	boost::shared_ptr<gpu_ptr_2D<float> > getD();

	/**
	 * Returns the gpu pointer to the eigenvalue-data on the GPU
	 * @see gpu_ptr_1D
	 * @return pointer to the GPU memory (NOT accessible by the CPU directly)
	 */
	boost::shared_ptr<gpu_ptr_1D<float> > getL();

	/**
	 * Returns the current version as a string.
	 * [major].[minor].[build]
	 * @return
	 */
	static const char* versionString();
	
	/**
	  * Returns the major version of the simulator
	  */
	static unsigned int majorVersion();

	/**
	  * Returns the minor version of the simulator
	  */
	static unsigned int minorVersion();

	/**
	  * Returns the build version of the simulator
	  */
	static unsigned int buildVersion();

	/**
	 * Returns the current bathymetry given as cell averages (NOT the bilinear input bathymetry)
	 * Remember that all offsets must include ghost cells
	 * @param cpu_data Data pointer to store values in. For speed, please use pinned memory
	 * 	allocated by cudaMallocHost. Must be pre-allocated to hold nx by ny values.
	 * @param nx_offset Offset into height map in number of cells
	 * @param ny_offset Offset into height map in number of cells
	 * @param nx Number of cells in x direction to read
	 * @param ny Number of cells in y direction to read
	 */
	void getB(float* cpu_data,
			unsigned int nx_offset=2, unsigned int ny_offset=2,
			unsigned int nx=0, unsigned int ny=0);

	/**
	 * Returns the current water depth (h) given as grid cell averages. Remember that all offsets must
	 * include ghost cells.
	 * @param cpu_data Data pointer to store values in. For speed, please use pinned memory
	 * 	allocated by cudaMallocHost. Must be pre-allocated to hold nx by ny values.
	 * @param nx_offset Offset into water map in number of cells
	 * @param ny_offset Offset into water map in number of cells
	 * @param nx Number of cells in x direction to read
	 * @param ny Number of cells in y direction to read
	 */
	void getU1(float* cpu_data,
			unsigned int nx_offset=2, unsigned int ny_offset=2,
			unsigned int nx=0, unsigned int ny=0);

	/**
	 * Returns the current U-discharge (hu) given as grid cell averages. Remember that all offsets must
	 * include ghost cells.
	 * @param cpu_data Data pointer to store values in. For speed, please use pinned memory
	 * 	allocated by cudaMallocHost. Must be pre-allocated to hold nx by ny values.
	 * @param nx_offset Offset into water map in number of cells
	 * @param ny_offset Offset into water map in number of cells
	 * @param nx Number of cells in x direction to read
	 * @param ny Number of cells in y direction to read
	 */
	void getU2(float* cpu_data,
			unsigned int nx_offset=2, unsigned int ny_offset=2,
			unsigned int nx=0, unsigned int ny=0);

	/**
	 * Returns the current V-discharge (hv) given as grid cell averages. Remember that all offsets must
	 * include ghost cells.
	 * @param cpu_data Data pointer to store values in. For speed, please use pinned memory
	 * 	allocated by cudaMallocHost. Must be pre-allocated to hold nx by ny values.
	 * @param nx_offset Offset into water map in number of cells
	 * @param ny_offset Offset into water map in number of cells
	 * @param nx Number of cells in x direction to read
	 * @param ny Number of cells in y direction to read
	 */
	void getU3(float* cpu_data,
			unsigned int nx_offset=2, unsigned int ny_offset=2,
			unsigned int nx=0, unsigned int ny=0);

	/**
	 * Sets the current water depths (h) given as grid cell averages.
	 * Remember that all offsets must include ghost cells.
	 * @param cpu_data Pointer to the values to be written.
	 * For speed, please use pinned memory allocated by cudaMallocHost.
	 * Must be pre-allocated to hold nx by ny values.
	 * @param nx_offset Offset into water map in number of cells
	 * @param ny_offset Offset into water map in number of cells
	 * @param nx Number of cells in x direction to write
	 * @param ny Number of cells in y direction to write
	 */
	void setU1(const float* cpu_data,
			unsigned int nx_offset=2, unsigned int ny_offset=2,
			unsigned int nx=0, unsigned int ny=0);

	/**
	 * Sets the current U-discharge (hu) given as grid cell averages.
	 * Remember that all offsets must include ghost cells.
	 * @param cpu_data Pointer to the values to be written.
	 * For speed, please use pinned memory allocated by cudaMallocHost.
	 * Must be pre-allocated to hold nx by ny values.
	 * @param nx_offset Offset into water map in number of cells
	 * @param ny_offset Offset into water map in number of cells
	 * @param nx Number of cells in x direction to write
	 * @param ny Number of cells in y direction to write
	 */
	void setU2(const float* cpu_data,
			unsigned int nx_offset=2, unsigned int ny_offset=2,
			unsigned int nx=0, unsigned int ny=0);

	/**
	 * Sets the current V-discharge (hv) given as grid cell averages.
	 * Remember that all offsets must include ghost cells.
	 * @param cpu_data Pointer to the values to be written.
	 * For speed, please use pinned memory allocated by cudaMallocHost.
	 * Must be pre-allocated to hold nx by ny values.
	 * @param nx_offset Offset into water map in number of cells
	 * @param ny_offset Offset into water map in number of cells
	 * @param nx Number of cells in x direction to write
	 * @param ny Number of cells in y direction to write
	 */
	void setU3(const float* cpu_data,
			unsigned int nx_offset=2, unsigned int ny_offset=2,
			unsigned int nx=0, unsigned int ny=0);

	/**
	 * Updates existing bilinear (NOT cell aveages) bathymetry. Simultaneously, it also computes
	 * the new value of the height map at grid cell centers affected by the update. Remember
	 * all offsets must include ghost cells.
	 * @param cpu_data new height map in the region. Must be nx by ny wide
	 * @param nx_offset Offset into the existing height map in number of intersections
	 * @param ny_offset Offset into the existing height map in number of intersections
	 * @param nx Number of intersections to update in x direction
	 * @param ny Number of intersections to update in y direction
	 */
	void setB(const float* cpu_data,
			unsigned int nx_offset=2, unsigned int ny_offset=2,
			unsigned int nx=0, unsigned int ny=0);

	/** 
	 * Enables use of the early exit strategy
	 */
	void enableEarlyExit();

	/**
	  * Disables use of the early exit strategy
	  */
	void disableEarlyExit();

	std::ostream& print(std::ostream& out) const;

protected:
	struct Impl;
	boost::shared_ptr<Impl> pimpl;
};






std::ostream& operator<<(std::ostream& out, const KPSimulator& sim);
std::ostream& operator<<(std::ostream& out, const KPSimulator* sim);






#endif /* KPSIMULATOR_H_ */
