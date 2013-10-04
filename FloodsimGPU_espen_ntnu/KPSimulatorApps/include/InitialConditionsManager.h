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

#ifndef INITIAL_CONDITIONS_MANAGER_H__
#define INITIAL_CONDITIONS_MANAGER_H__

#include "KPInitialConditions.h"
#include "datatypes.h"
#include <boost/program_options/variables_map.hpp>
#include <stdexcept>

class InitialConditionsManager {
public:
	InitialConditionsManager(boost::program_options::variables_map& map);
	inline KPInitialConditions getIC() { 
		try {
			KPInitialConditions init(params);
			std::cout << init << std::endl;
			return init; 
		}
		catch (std::runtime_error e) {
			std::cout << "Error getting initial conditions: " << std::endl;
			std::cout << e.what() << std::endl;
			exit(-1);
		}
	}

private:
	static boost::shared_ptr<Field> getB(boost::program_options::variables_map& map);
	static boost::shared_ptr<Field> getU1(boost::program_options::variables_map& map);
	static boost::shared_ptr<Field> getU2(boost::program_options::variables_map& map);
	static boost::shared_ptr<Field> getU3(boost::program_options::variables_map& map);
	static boost::shared_ptr<Field> getM(boost::program_options::variables_map& map);
	static KPBoundaryCondition getBC(std::string name, boost::program_options::variables_map& map);

	KPInitialConditions::KPParameters getParams(boost::program_options::variables_map& map);

private:
	KPInitialConditions::KPParameters params;
	boost::shared_ptr<Field> B, U1, U2, U3, M;
};

#endif