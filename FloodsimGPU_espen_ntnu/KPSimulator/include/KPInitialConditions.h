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

#ifndef KPINITIALCONDITIONS_H_
#define KPINITIALCONDITIONS_H_

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
#include <limits>
#include <iostream>
#include <sstream>
#include <vector>
#include <boost/utility.hpp>
#include <cassert>
#include <cuda_runtime.h>

#include "KPBoundaryConditions.h"

/**
 * Class that describes initial conditions we need to set up a simulation.
 * This object owns no data, and only keeps pointers to data, and other simulation
 * initial conditions.
 * FIXME: Document this struct thoroughly
 */
class KPInitialConditions {
public:
	/**
	 * Definition of different time ODE integrators
	 */
	enum TIME_INTEGRATOR {
		EULER=0,            //!< Euler integration
		RUNGE_KUTTA_2=1,    //!< Second order accurate Runge-Kutta integration
		AUTO=2,
		INTEGRATOR_UNKNOWN, //!< INTEGRATOR_UNKNOWN
	};

	struct ManningCoefficient {
		ManningCoefficient() {
			spatially_varying = false;
			n = NULL;
		}
		bool spatially_varying;
		float* n;
	};

	struct KPParameters {
		KPParameters() :
			B(NULL), 
			U1(NULL),
			U2(NULL),
			U3(NULL),
			M(),
			ode(INTEGRATOR_UNKNOWN),
			bc_north(),
			bc_south(),
			bc_east(),
			bc_west(),
			nx(0),
			ny(0),
			dx(0.0f),
			dy(0.0f),
			dt_scale(1.0f),
			g(9.80665f)
            {};

		float* B;                 //!< Bathymetry
		float* U1;                //!< Water elevation
		float* U2;                //!< hu discharge
		float* U3;                //!< hv discharge

		ManningCoefficient M;          //!< Manning friction coefficient
		TIME_INTEGRATOR ode;           //!< Time integrator

		KPBoundaryCondition bc_north;  //!< Boundary condition for north boundary
		KPBoundaryCondition bc_south;  //!< Boundary condition for north boundary
		KPBoundaryCondition bc_east;   //!< Boundary condition for north boundary
		KPBoundaryCondition bc_west;   //!< Boundary condition for north boundary

		unsigned int nx, ny;           //!< Number of initial grid cell intersections for bathymetry
		float dx, dy;                  //!< Grid cell spacing
		float dt_scale;                //!< Scaling of dt to ensure stability
		float g;                       //!< Gravitational constant.
	};

public:
	/**
	 * Constructor that sets all required variables for a simulation.
	 */
	KPInitialConditions(KPParameters params);

	KPInitialConditions() {
		throw "ERROR IN KPINITIALCONDITIONS";
	}

	virtual ~KPInitialConditions();

	/**
	 * @return Pointer to the bathymetry
	 */
	inline float* getB() const { return params.B; }

	/**
	 * @return Pointer to water elevation
	 */
	inline float* getU1() const { return params.U1; }

	/**
	 * @return Pointer to water discharge in u-direction
	 */
	inline float* getU2() const { return params.U2; }

	/**
	 * @return Pointer to water discharge in v-direction
	 */
	inline float* getU3() const { return params.U3; }

	/**
	 * @return Pointer to spatially varying Manning friction coefficient
	 */
	inline KPInitialConditions::ManningCoefficient getM() const { return params.M; }

	/**
	 * @return The time integrator type to use
	 */
	inline TIME_INTEGRATOR getTimeIntegrator() const { return params.ode; }

	/**
	 * @return Number of values for the bathymetry pointer. This should always return
	 * (nx+1) for a nx wide domain.
	 */
	inline unsigned int getBNx() const { return params.nx+1; }


	/**
	 * @return Number of values for the bathymetry pointer. This should always return
	 * (ny+1) for a ny wide domain.
	 */
	inline unsigned int getBNy() const { return params.ny+1; }

	/**
	 * @return Number of values for the U-pointers
	 */
	inline unsigned int getUNx() const { return params.nx; }

	/**
	 * @return Number of values for the U-pointers
	 */
	inline unsigned int getUNy() const { return params.ny; }

	/**
	 * @return Width of domain in number of cells
	 */
	inline const unsigned int& getNx() const { return params.nx; }

	/**
	 * @return Height of domain in number of cells
	 */
	inline const unsigned int& getNy() const { return params.ny; }

	/**
	 * @return Grid cell spacing in x direction
	 */
	inline const float& getDx() const { return params.dx; }

	/**
	 * @return Grid cell spacing in y direction
	 */
	inline const float& getDy() const { return params.dy; }

	/**
	 * @return Scaling factor of Dt to ensure stability
	 */
	inline const float& getDtScale() const { return params.dt_scale; }
    
	/**
	 * @return Gravitational constant
	 */
	inline const float& getG() const { return params.g; }
    	
	/**
	 * @return Boundary conditions for north interface
	 * @see KPBoundaryCondition
	 */
	inline const KPBoundaryCondition& getNorthBC() const { return params.bc_north; }
	
	/**
	 * @return Boundary conditions for south interface
	 * @see KPBoundaryCondition
	 */
	inline const KPBoundaryCondition& getSouthBC() const { return params.bc_south; }
	
	/**
	 * @return Boundary conditions for east interface
	 * @see KPBoundaryCondition
	 */
	inline const KPBoundaryCondition& getEastBC() const { return params.bc_east; }
	
	/**
	* @return Boundary conditions for west interface
	* @see KPBoundaryCondition
	 */
	inline const KPBoundaryCondition& getWestBC() const { return params.bc_west; }
    
private:
	/**
	 * Checks whether these initial conditions are valid or not.
	 */
	void sanityCheck() const;

protected:
	KPParameters params;
};




/**
 * Helper function to easily print out time integrator
 */
inline std::ostream& operator<<(std::ostream& out, const KPInitialConditions::TIME_INTEGRATOR& t) {
	switch(t) {
	case KPInitialConditions::EULER: out << "Euler"; break;
	case KPInitialConditions::RUNGE_KUTTA_2: out << "2nd order TVD Runge-Kutta"; break;
	default: out << "Unknown";
	}
	return out;
}




/**
 * Helper function to easily print out manning coefficient
 */
inline std::ostream& operator<<(std::ostream& out, const KPInitialConditions::ManningCoefficient& M) {
	if (M.spatially_varying) {
		out << "[" << M.n[0] << ", " << M.n[1] << ", ...]";
	}
	else {
		out << M.n[0];
	}
	return out;
}

/**
 * Helper function to easily print out initial conditions
 */
inline std::ostream& operator<<(std::ostream& out, const KPInitialConditions& init) {
	out << "Domain size: [" << init.getNx() << "x" << init.getNy() << "], ";
	out << "cell size: [" << init.getDx() << "x" << init.getDy() << "]" << std::endl;
	out << "g = " << init.getG() << ", Mannings n = " << init.getM() << std::endl;
	out << "Boundary conditions: ";
	out << "[n=" << init.getNorthBC().getType() 
		<< ", s=" << init.getSouthBC().getType()
		<< ", e=" << init.getEastBC().getType()
		<< ", w=" << init.getWestBC().getType() << std::endl;
	out << "Time integrator: " << init.getTimeIntegrator() << ", ";
	out << "Dt scaled by " << init.getDtScale() << ", " << std::endl;
	return out;
}



/**
 * Helper function to easily print out initialconditions
 */
inline std::ostream& operator<<(std::ostream& out, const KPInitialConditions* init) {
	out << *init;
	return out;
}











#endif /* KPINITIALCONDITIONS_H_ */
