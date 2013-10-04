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

#include "InitialConditionsManager.h"

#include "FileManager.h"
#include "DataManager.h"

#include <sstream>

#include "app/data_generation.hpp"

using boost::shared_ptr;

InitialConditionsManager::InitialConditionsManager(boost::program_options::variables_map& map) {
	params = getParams(map);
}

shared_ptr<Field> InitialConditionsManager::getB(boost::program_options::variables_map& map) {
	std::stringstream log;

	if (map.count("bathymetry_no") && map.count("nx") && map.count("ny")) {
		int nx = map["nx"].as<int>();
		int ny = map["ny"].as<int>();
		return generate_bathymetry(map["bathymetry_no"].as<int>(), nx, ny);
	}
	if (map.count("bathymetry") && map.count("filetype"))
		return FileManager::readFile(map["bathymetry"].as<std::string>(), map["filetype"].as<std::string>());

	log << "Could not get terrain. Try the following:" << std::endl;
	log << "* specify 'bathymetry' and 'filetype'," << std::endl;
	log << "* specify 'bathymetry_no', 'water_elevation_no', 'nx', 'ny', 'dx' and 'dy'" << std::endl;
	log << "* or try --help." << std::endl;
	throw std::runtime_error(log.str());
}

shared_ptr<Field> InitialConditionsManager::getU1(boost::program_options::variables_map& map) {
	std::stringstream log;

	if (map.count("water_elevation_no") && map.count("nx") && map.count("ny")) {
		int nx = map["nx"].as<int>();
		int ny = map["ny"].as<int>();
		return generate_water_elevation(map["water_elevation_no"].as<int>(), nx, ny);
	}
	if (map.count("water_elevation") && map.count("filetype"))
		return FileManager::readFile(map["water_elevation"].as<std::string>(), map["filetype"].as<std::string>());

	log << "Could not get water elevation. Try the following:" << std::endl;
	log << "* specify 'water_elevation' and 'filetype'," << std::endl;
	log << "* specify 'terrain_no', 'nx' and 'ny'," << std::endl;
	log << "* or try --help." << std::endl;
	throw std::runtime_error(log.str());
}

shared_ptr<Field> InitialConditionsManager::getU2(boost::program_options::variables_map& map) {
	if (map.count("u_discharge") && map.count("filetype"))
		return FileManager::readFile(map["u_discharge"].as<std::string>(), map["filetype"].as<std::string>());
	else if (map.count("u_discharge_no") && map.count("nx") && map.count("ny")) {
		int nx = map["nx"].as<int>();
		int ny = map["ny"].as<int>();
		return generate_u_discharge(map["u_discharge_no"].as<int>(), nx, ny);
	}
	else {
		return boost::shared_ptr<Field>(new Field());
	}
}

shared_ptr<Field> InitialConditionsManager::getU3(boost::program_options::variables_map& map) {
	if (map.count("v_discharge") && map.count("filetype"))
		return FileManager::readFile(map["v_discharge"].as<std::string>(), map["filetype"].as<std::string>());
	else if (map.count("v_discharge_no") && map.count("nx") && map.count("ny")) {
		int nx = map["nx"].as<int>();
		int ny = map["ny"].as<int>();
		return generate_v_discharge(map["v_discharge_no"].as<int>(), nx, ny);
	}
	else {
		return boost::shared_ptr<Field>(new Field());
	}
}

shared_ptr<Field> InitialConditionsManager::getM(boost::program_options::variables_map& map) {
	if (map.count("manning_coefficient") && map.count("filetype")) {
		std::string manning_coefficient = map["manning_coefficient"].as<std::string>();
		if (FileManager::fileExists(manning_coefficient)) {
			return FileManager::readFile(map["manning_coefficient"].as<std::string>(), map["filetype"].as<std::string>());
		}
		else {
			boost::shared_ptr<Field> data;
			data.reset(new Field(1, 1));

			std::istringstream ss(manning_coefficient);
			ss >> std::dec >> data->data[0];

			if (ss.fail()) {
				std::cout << "Unable to parse " << manning_coefficient << " as a file or number." << std::endl;
				exit(-1);
			}
			else {
				return data;
			}
		}
	}
	else if (map.count("manning_coefficient_no") && map.count("nx") && map.count("ny")) {
		int nx = map["nx"].as<int>();
		int ny = map["ny"].as<int>();
		return generate_manning_coefficient(map["manning_coefficient_no"].as<int>(), nx, ny);
	}
	else {
		boost::shared_ptr<Field> data;
		data.reset(new Field(1, 1));
		data->data[0] = 0.0f;
		return data;
	}
}


KPBoundaryCondition InitialConditionsManager::getBC(std::string name, boost::program_options::variables_map& map) {
	std::string bc_name = "bc_" + name;
	std::string name_values = bc_name + "_values";
	std::string name_value = bc_name + "_value";

	KPBoundaryCondition bc(static_cast<KPBoundaryCondition::TYPE>(map[bc_name].as<int>()));

	if (bc.getType() == KPBoundaryCondition::FIXED_DEPTH || bc.getType() == KPBoundaryCondition::FIXED_DISCHARGE) {
		if (map.count(name_values))
			FileManager::readBCValues(map[name_values].as<std::string>(), bc.getTimes(), bc.getValues());
		else if (map.count(name_value)) {
			bc.getTimes().at(0) = 0.0f;
			bc.getValues().at(0) = map[name_value].as<float>();
		}
		else {
			std::stringstream log;
			log << "You must supply at least one value for " << name << " boundary condition" << std::endl;
			throw std::runtime_error(log.str());
		}
	}

	return bc;
}

KPInitialConditions::KPParameters InitialConditionsManager::getParams(boost::program_options::variables_map& map) {
	KPInitialConditions::KPParameters params;
	using std::string;
	using std::min;
	using std::max;

	B = getB(map);
	U1 = getU1(map);
	U2 = getU2(map);
	U3 = getU3(map);
	M = getM(map);

	params.bc_north = getBC("north", map);
	params.bc_south = getBC("south", map);
	params.bc_east = getBC("east", map);
	params.bc_west = getBC("west", map);

	//Get number of grid cells
	params.nx = B->nx-1;
	params.ny = B->ny-1;

	//Perform post-processing of the data to give us a better input data source...
	float dz = map["dz"].as<float>();
	std::cout << "Post processing B..." << std::endl;
	DataManager::postProcess(B, DataManager::FILL, dz);
	std::cout << "Post processing U1..." << std::endl;
	DataManager::postProcess(U1, DataManager::FILL, dz);
	std::cout << "Post processing M..." << std::endl;
	DataManager::postProcess(M, DataManager::ZERO);

#ifndef NDEBUG
	std::cout << "Input dimensions" << std::endl;
	std::cout << "U1[" << U1->nx << "x" << U1->ny << "]" << std::endl;
	std::cout << "U2[" << U2->nx << "x" << U2->ny << "]" << std::endl;
	std::cout << "U3[" << U3->nx << "x" << U3->ny << "]" << std::endl;
	std::cout << "B[" << B->nx << "x" << B->ny << "]" << std::endl;
	std::cout << "M[" << M->nx << "x" << M->ny << "]" << std::endl;
#endif

	//Try to get U1-U3 placed at the correct locations
	if (U1->nx != params.nx) {
		std::cout << "Warning: U1 dimensions incorrect. Attempting bilinear intersections to centers filtering" << std::endl;
		DataManager::intersectionsToCenters(U1);
	}
	if (U2->nx * U2->ny > 0 && U2->nx != params.nx) {
		std::cout << "Warning: U2 dimensions incorrect. Attempting bilinear intersections to centers filtering" << std::endl;
		DataManager::intersectionsToCenters(U2);
	}
	if (U3->nx * U3->ny > 0 && U3->nx != params.nx) {
		std::cout << "Warning: U3 dimensions incorrect. Attempting bilinear intersections to centers filtering" << std::endl;
		DataManager::intersectionsToCenters(U3);
	}
	if (M->nx * M->ny != 1 && M->nx != params.nx) {
		std::cout << "Warning: M dimensions incorrect. Attempting bilinear intersections to centers filtering" << std::endl;
		DataManager::intersectionsToCenters(M);
	}

	std::cout << "Input variable dimensions" << std::endl;
	std::cout << "U1[" << U1->nx << "x" << U1->ny << "]" << std::endl;
	std::cout << "U2[" << U2->nx << "x" << U2->ny << "]" << std::endl;
	std::cout << "U3[" << U3->nx << "x" << U3->ny << "]" << std::endl;
	std::cout << "B[" << B->nx << "x" << B->ny << "]" << std::endl;
	std::cout << "M[" << M->nx << "x" << M->ny << "]" << std::endl;

	//Actually test that we have the correct dimensions
	if (	U1->nx != params.nx || U1->ny != params.ny ||
		(U2->data != NULL && (U2->nx != params.nx || U2->ny != params.ny)) ||
		(U3->data != NULL && (U3->nx != params.nx || U3->ny != params.ny)) ||
		B->nx != params.nx+1 || B->ny != params.ny+1 ||
		((M->nx != params.nx || M->ny != params.ny) && M->nx*M->ny != 1)) {
			std::cout << "Wrong dimension for U, B, or M:" << std::endl;
			std::cout << "U1[" << U1->nx << "x" << U1->ny << "]" << std::endl;
			if (U2->data != NULL) std::cout << "U2[" << U2->nx << "x" << U2->ny << "]" << std::endl;
			if (U3->data != NULL) std::cout << "U3[" << U3->nx << "x" << U3->ny << "]" << std::endl;
			std::cout << "B[" << B->nx << "x" << B->ny << "]" << std::endl;
			std::cout << "M[" << M->nx << "x" << M->ny << "]" << std::endl;
			exit(-1);
	}

	//Get grid cell spacing
	if (B->dx == -1 && B->dy == -1) {
		B->dx = U1->dx;
		B->dy = U1->dy;
	}
	else if (U1->dx == -1 && U1->dy == -1) {
		U1->dx = B->dx;
		U1->dy = B->dy;
	}
	else if (B->dx != U1->dx || B->dy != U1->dy) {
		std::cout << "Dx and Dy do not match up between B and U1:" << std::endl;
		std::cout << "U[" << U1->dx << "x" << U1->dy << "]" << std::endl;
		std::cout << "B[" << B->dx << "x" << B->dy << "]" << std::endl;
		exit(-1);
	}
	params.dx = B->dx;
	params.dy = B->dy;

	if (map.count("dx"))     params.dx = map["dx"].as<float>();
	if (map.count("dy"))     params.dy = map["dy"].as<float>();
	if (map.count("width"))  params.dx = map["width"].as<float>()/(float) params.nx;
	if (map.count("height")) params.dy = map["height"].as<float>()/(float) params.ny;

	if (params.dx <= 0 || params.dy <= 0) {
		std::cout << "Invalid dx or dy." << std::endl;
		exit(-1);
	}

	params.B = B->data;
	params.U1 = U1->data;
	params.U2 = U2->data;
	params.U3 = U3->data;
	params.M.n = M->data;
	params.M.spatially_varying = (M->nx * M->ny != 1);
	params.ode = static_cast<KPInitialConditions::TIME_INTEGRATOR>(map["time_integrator"].as<int>());

	//Set misc parameters
	if (map.count("gravitational_constant")) params.g = map["gravitational_constant"].as<float>();
	if (map.count("dt_scale"))               params.dt_scale = map["dt_scale"].as<float>();

	return params;
}
