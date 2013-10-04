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

#include "KPInitialConditions.h"

#include <stdexcept>

KPInitialConditions::KPInitialConditions(KPParameters params_) {
	params = params_;
	
	sanityCheck();
}

KPInitialConditions::~KPInitialConditions() { }

void KPInitialConditions::sanityCheck() const {
	std::stringstream log;
	bool error = false;

	log << "Runtime error in KPInitialConditions::sanityCheck()" << std::endl;
	if (params.nx <= 0) {
		log << "invalid nx: " << params.nx << std::endl;
		error = true;
	}

	if (params.ny <= 0) {
		log << "invalid ny: " << params.ny << std::endl;
		error = true;
	}

	if (params.dx <= 0) {
		log << "invalid dx: " << params.dx << std::endl;
		error = true;
	}
	if (params.dy <= 0) {
		log << "invalid dy: " << params.dy << std::endl;
		error = true;
	}

	if (params.B == NULL) {
		log << "Invalid B pointer: " << params.B << std::endl;
		error = true;
	}

	if (params.U1 == NULL) {
		log << "Invalid U1 pointer: " << params.U1 << std::endl;
		error = true;
	}

	if (params.dt_scale <= 0.0f) {
		log << "Invalid dt scaling: " << params.dt_scale << std::endl;
		error = true;
	}

	if (params.ode == INTEGRATOR_UNKNOWN) {
		log << "Invalid time integrator: " << params.ode << std::endl;
		error = true;
	}

	if (params.bc_north.getType() == KPBoundaryCondition::UNKNOWN) {
		log << "Invalid north boundary conditions: " << params.bc_north << std::endl;
		error = true;
	}

	if (params.bc_south.getType() == KPBoundaryCondition::UNKNOWN) {
		log << "Invalid south boundary conditions: " << params.bc_south << std::endl;
		error = true;
	}

	if (params.bc_east.getType() == KPBoundaryCondition::UNKNOWN) {
		log << "Invalid east boundary conditions: " << params.bc_east << std::endl;
		error = true;
	}

	if (params.bc_west.getType() == KPBoundaryCondition::UNKNOWN) {
		log << "Invalid west boundary conditions: " << params.bc_west << std::endl;
		error = true;
	}

	if (params.M.n == NULL) {
		log << "Manning coefficient not allocated!" << std::endl;
		error = true;
	}

	if (params.M.spatially_varying) {
		for (unsigned int j=0; j<params.ny; ++j) {
			for (unsigned int i=0; i<params.nx; ++i) {
				const float value = params.M.n[j*params.nx+i];
				if (value < 0.0f) {
					log << "Invalid  manning coefficient at [" 
						<< i << ", " << j
						<< "] : " << value << std::endl;
					error = true;
				}
			}
		}
	}
	else {
		if (params.M.n[0] < 0.0f) {
			log  << "Invalid manning coefficient : " << params.M.n[0] << std::endl;
			error = true;
		}
	}

	if (params.g <= 0.0f) {
		log << "Invalid  gravitational constant: " << params.g << std::endl;
		error = true;
	}
    
	if (error) {
		throw std::runtime_error(log.str());
	}
}