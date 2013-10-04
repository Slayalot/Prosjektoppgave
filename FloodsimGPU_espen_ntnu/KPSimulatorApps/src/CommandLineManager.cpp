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

#include "CommandLineManager.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>




using namespace boost::program_options;
using namespace std;

CommandLineManager::CommandLineManager(options_description& input_options) 
			: options("Most frequently used options") {
	addMainOptions();
	addODEOptions();
	addBCOptions();
	addSourceOptions();
	addICOptions();
	options.add(input_options);
	positional_options.add("config", 1);
}

variables_map CommandLineManager::getVariablesMap(int argc, char** argv) {
	//Create commandline parser
	command_line_parser cli_parser(argc, argv);
	cli_parser.positional(positional_options);
	cli_parser.options(options);
	cli_parser.allow_unregistered();

	//Parse, and store in map
	parsed_options cli_po = cli_parser.run();
	store(cli_po, cli_vars);

	printUnrecognized(cli_po);


	if (cli_vars.count("config")) {
		string config_file = cli_vars["config"].as<string>();
		ifstream ifs(config_file.c_str());
		if (ifs.good()) {
			parsed_options cf_po = parse_config_file(ifs, options, true);
			store(cf_po, cli_vars);

			printUnrecognized(cf_po);
		}
		else {
			stringstream log;
			log << "Could not open config file '" << config_file << "'" << endl;
			throw std::runtime_error(log.str());
		}
	}
	notify(cli_vars);
#ifndef NDEBUG
	printVars(cli_vars);
#endif

	return cli_vars;
}

bool CommandLineManager::validate() {
	//FIXME: Validate thoroughly here

	if (cli_vars.count("help")) {
		cout << "Usage: <program> [options] <config-file>" << endl;
		cout << options << endl;
		return false;
	}

	return false;
}

variables_map cli_vars;
void CommandLineManager::addMainOptions() {
	options.add_options()
		("help,h", "produce help message")
		("filetype,t", value<string>()->default_value("dem"), "Filetype of bathymetry and water elevation")
        ("config,c", value<string>(), "Configuration filename")
        ("alternate_flux", value<bool>()->default_value(false), "Use alternate flux formulation?")
		("bathymetry,b", value<string>(), "Bathymetry filename")
		("water_elevation,w", value<string>(), "Water elevation filename")
		("u_discharge,u", value<string>(), "U discharge filename")
		("v_discharge,v", value<string>(), "V discharge filename")
		("manning_coefficient,n", value<string>(), "Mannings n")
		("gravitational_constant,g", value<float>(), "Gravitational acceleration coefficient")
    	("debug_dump_input", value<bool>(), "Dump input data as received by the simulator?");
}

void CommandLineManager::addODEOptions() {
	options_description ode_options("Time integration options");
	ode_options.add_options()
		("time_integrator", value<int>()->default_value(1), "Time integrator. 0=euler, 1=runge-kutta 2")
		("desingularization_eps", value<float>(), "Epsilon used to desingularize flux calculations")
		("dt_scale", value<float>(), "Scaling of Dt");
	options.add(ode_options);
}

void CommandLineManager::addBCOptions() {
	options_description bc_options("Boundary contition options");
	bc_options.add_options()
		("bc_north", value<int>()->default_value(1), "North boundary condition: 0=none, 1=wall, 2=fixed elevation, 3=fixed discharge, 4=open")
		("bc_south", value<int>()->default_value(1), "South boundary condition")
		("bc_east", value<int>()->default_value(1), "East boundary condition")
		("bc_west", value<int>()->default_value(1), "West boundary condition")
		("bc_north_value", value<float>(), "North boundary condition value")
		("bc_south_value", value<float>(), "South boundary condition value")
		("bc_east_value", value<float>(), "East boundary condition value")
		("bc_west_value", value<float>(), "West boundary condition value")
		("bc_north_values", value<string>(), "North boundary condition values")
		("bc_south_values", value<string>(), "South boundary condition values")
		("bc_east_values", value<string>(), "East boundary condition values")
		("bc_west_values", value<string>(), "West boundary condition values");
	options.add(bc_options);
}

void CommandLineManager::addICOptions() {
	options_description ic_options("Options changing generation of bathymetry and water");
	ic_options.add_options()
		("bathymetry_no", value<int>(), "Bathymetry number to generate")
		("water_elevation_no", value<int>(), "Water elevation number to generate")
		("u_discharge_no", value<int>()->default_value(0), "Water discharge number to generate")
		("v_discharge_no", value<int>()->default_value(0), "Water discharge number to generate")
		("manning_coefficient_no", value<int>(), "Friction coefficient to generate")
		("nx", value<int>(), "Number of grid cells to generate (when generating terrain)")
		("ny", value<int>(), "Number of grid cells to generate (when generating terrain)")
		("width", value<float>(), "Override grid cell spacing in x direction (dx=nx/height)")
		("height", value<float>(), "Override grid cell spacing in y direction")
		("dz", value<float>()->default_value(1.0f), "Set vertical scaling of terrain to generate")
		("dx", value<float>(), "Override grid cell spacing in x direction")
		("dy", value<float>(), "Override grid cell spacing in y direction");
	options.add(ic_options);
}

void CommandLineManager::addSourceOptions() {
	options_description source_options("Internal source and sink options");
	source_options.add_options()
		("source", value<vector<string> >(), "File that describes an internal source or sink");
	options.add(source_options);
}

void CommandLineManager::printVars(variables_map &cli_vars) {
	const unsigned int field_one_width = 25;
	const unsigned int field_two_width = 10;
	cout << "Options on command line:" << endl;
	for (variables_map::iterator it=cli_vars.begin(); it!=cli_vars.end(); ++it) {
		stringstream tmp;
		cout << setw(field_one_width) << left << it->first;
		bool success=false;

		if (success==false) {
			try {
				tmp << setw(field_two_width) << right << "[string] '" << it->second.as<string>();
				success=true;
			}
			catch(const boost::bad_any_cast &) {
				success=false;
			}
		}

		if (success==false) {
			try {
				tmp << setw(field_two_width) << right << "[float] '" << it->second.as<float>();
				success=true;
			}
			catch(const boost::bad_any_cast &) {
				success=false;
			}
		}

		if (success==false) {
			try {
				tmp << setw(field_two_width) << right << "[int] '" << it->second.as<int>();
				success=true;
			}
			catch(const boost::bad_any_cast &) {
				success=false;
			}
		}

		if (success==false) {
			try {
				tmp << setw(field_two_width) << right << "[bool] '" << it->second.as<bool>();
				success=true;
			}
			catch(const boost::bad_any_cast &) {
				success=false;
			}
		}

		if (success==false) {
			try {
				std::vector<int> vec = it->second.as<vector<int> >();
				tmp << setw(field_two_width) << right << "<int> '[";
				for (unsigned int i=0; i<vec.size(); ++i) {
					if (i>0) tmp << ", ";
					tmp << "'" << vec.at(i) << "'";
				}
				tmp << "]";
				success = true;
			}
			catch (const boost::bad_any_cast &) {
				success = false;
			}
		}

		if (success==false) {
			try {
				std::vector<string> vec = it->second.as<vector<string> >();
				tmp << setw(field_two_width) << right << "<string> '[";
				for (unsigned int i=0; i<vec.size(); ++i) {
					if (i>0) tmp << ", ";
					tmp << "'" << vec.at(i) << "'";
				}
				tmp << "]";
				success = true;
			}
			catch (const boost::bad_any_cast &) {
				success = false;
			}
		}

		if (success==false) {
			try {
				std::vector<float> vec = it->second.as<vector<float> >();
				tmp << setw(field_two_width) << right << "<float> '[";
				for (unsigned int i=0; i<vec.size(); ++i) {
					if (i>0) tmp << ", ";
					tmp << "'" << vec.at(i) << "'";
				}
				tmp << "]";
				success = true;
			}
			catch (const boost::bad_any_cast &) {
				success = false;
			}
		}

		if (success==false) {
			tmp << setw(field_two_width) << right << "{UNKNOWN} " << "!!!";
		}

		cout << tmp.str() << "'" << endl;
	}
}

void CommandLineManager::printUnrecognized(parsed_options &cli_po) {
	vector<string> unrecognized = collect_unrecognized(cli_po.options, exclude_positional);

	for (unsigned int i=0; i<unrecognized.size(); ++i) {
		std::cout << "Warning: Unrecognized option '" << unrecognized.at(i) << "'. Ignoring..." << std::endl;
	}
}
