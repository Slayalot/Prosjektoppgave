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

#include <iostream>
#include <string>
#include <iomanip>
#include <sstream>
#include <limits>

#include "KPSimulator.h"
#include "configure.h"
#include "CommandLineManager.h"
#include "InitialConditionsManager.h"
#include "FileManager.h"
#include "app/common.hpp"
#ifdef KPSIMULATORAPPS_USE_NETCDF
//#include "NetCDFHandler.h"
#include "KPNetCDFFile.h"
#endif
#include "util.hpp"

using std::endl;
using std::fixed;
using std::scientific;
using std::cout;
using std::flush;
using std::ifstream;
using std::right;
using std::string;
using std::max;
using std::setprecision;
using std::stringstream;
using std::setw;
using std::setfill;

using boost::shared_ptr;
using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;


int main(int argc, char** argv) {
	float total_time, write_every, next_dump_time = 0.0f, next_print_time=0.0f;
	shared_ptr<KPSimulator> sim;
	string output_file;
	double start, end;
	bool write_to_file = false;
#ifdef KPSIMULATORAPPS_USE_NETCDF
	shared_ptr<KPNetCDFFile> file;
	shared_ptr<TimeStep> buffer;
#else
	shared_ptr<Field> buffer;
#endif
	bool hotstart = false;
	bool check_conservation = false;



	//Parse command line options and generate initial conditions from it
    options_description options("Misc options");
	options.add_options()
#ifdef KPSIMULATORAPPS_USE_NETCDF
				("hotstart", value<bool>()->default_value(false)->zero_tokens(), "Enable hotstarting simulation by using output NetCDF file as initial conditions?")
#endif
				("total_time", value<float>(), "Total simulation time")
				("write_every", value<float>()->default_value(1), "Write every t (simulation) seconds to disk")
				("check_conservation", value<bool>()->zero_tokens(), "Check conservation of mass")
				("output,o", value<string>(), "Output filename");
	CommandLineManager cli(options);
	variables_map cli_vars = cli.getVariablesMap(argc, argv);
	if (!cli_vars.count("total_time")) {
		cout << "Missing 'total_time', try --help" << endl;
		exit(-1);
	}
	if (cli_vars.count("output") && cli_vars["output"].as<string>().compare("") != 0) {
		write_to_file = true;
		output_file = cli_vars["output"].as<string>();
#ifdef KPSIMULATORAPPS_USE_NETCDF
		hotstart = cli_vars["hotstart"].as<bool>();
#endif
	}
	write_every = cli_vars["write_every"].as<float>();
	total_time = cli_vars["total_time"].as<float>();
	if (cli_vars.count("check_conservation")) check_conservation = cli_vars["check_conservation"].as<bool>();



	// Create simulator
	if (FileManager::fileExists(output_file)) {
#ifdef KPSIMULATORAPPS_USE_NETCDF
		if (hotstart) {
			std::cout << "Hotstart: Continuing simulation in " << output_file << ". ";
			std::cout << "Parameters given on command line ignored" << std::endl;
		}
		else {
			std::cout << "Attempting to overwrite existing NetCDF file. Aborting. (" << output_file << ")" << std::endl;
			exit(-1);
		}
		shared_ptr<Field> B;
		shared_ptr<Field> M;
		shared_ptr<TimeStep> ts;
		file.reset(new KPNetCDFFile(output_file));
		KPInitialConditions init = file->readInitialConditions(B, M, ts);
		buffer.reset(new TimeStep(init.getNx(), init.getNy()));
		sim.reset(new KPSimulator(init));
#endif
	}
	else {
		InitialConditionsManager ic_manager(cli_vars);
		KPInitialConditions init = ic_manager.getIC();
		sim.reset(new KPSimulator(init));

		// Open output file stuff
		if (write_to_file) {
#ifdef KPSIMULATORAPPS_USE_NETCDF
			file.reset(new KPNetCDFFile(output_file));
			buffer.reset(new TimeStep(init.getNx(), init.getNy()));
			file->writeInitialConditions(init);
#endif
		}
		if (check_conservation) {
#ifdef KPSIMULATORAPPS_USE_NETCDF
			buffer.reset(new TimeStep(init.getNx(), init.getNy()));
#else
			buffer.reset(new Field(init.getNx(), init.getNy()));
#endif
		}
	}

	cout	<< setw(7) << right << "n" 
			<< ", " << setw(8) << right << "t_s" 
			<< ", " << setw(8) << right << "t_w" 
			<< ", " << setw(8) << right << "dt" 
			<< ", " << setw(7) << right << "ips" 
			<< ", " << setw(8) << right << "t_w_rem" 
			<< ", " << setw(7) << right << "~%";
	if (check_conservation) {
		cout << ", " << setw(9) << right << "mass";
	}
	cout << endl << fixed;

	start = getCurrentTime();
	end = start;
	while (sim->getTime() < total_time) {
		while (sim->getTime() < next_dump_time && (end - start) < next_print_time) {
			sim->step();
			end = getCurrentTime();
		}

		//Print out running stats
		if (end-start >= next_print_time || sim->getTime() >= next_dump_time) {
			float timestep = static_cast<float>(sim->getTimeSteps());
			float simulation_time = sim->getTime();
			float percent = 100.0f * simulation_time / total_time;
			float wall_time = static_cast<float>(end-start);
			float iterations_per_second = timestep / wall_time;
			float avg_dt = simulation_time / timestep;
			float timesteps_remaining = max((total_time - simulation_time) / avg_dt, 0.0f);
			float est_wall_time_remaining = max(timesteps_remaining / iterations_per_second, 0.0f);
			
			cout << setw(7) << setprecision(0) << timestep 
					<< ", " << right << printTime(simulation_time) 
					<< ", " << right << printTime(wall_time)
					<< ", " << setw(8) << setprecision(2) << right << scientific << sim->getDt() << fixed 
					<< ", " << setw(7) << setprecision(1) << right << iterations_per_second 
					<< ", " << right << printTime(est_wall_time_remaining) 
					<< ", " << setw(6) << setprecision(0) << right << percent << "%";
			next_print_time = static_cast<float>(end) + 0.5f;

			if (check_conservation) {
				double water_mass = 0.0;

#ifdef KPSIMULATORAPPS_USE_NETCDF
				sim->getU1(buffer->U[0]->data);
#else
				sim->getU1(buffer->data);
#endif

#pragma omp parallel for
				for (int i=0; i<static_cast<int>(sim->getIC().getNx()*sim->getIC().getNy()); ++i) {
#ifdef KPSIMULATORAPPS_USE_NETCDF
					double w = static_cast<double>(buffer->U[0]->data[i]);
#else
					double w = static_cast<double>(buffer->data[i]);
#endif
					water_mass += w;
				}
				cout << ", " << setw(9) << setprecision(4) << scientific << right << water_mass*sim->getIC().getDx()*sim->getIC().getDy() << fixed;
			}
			
			cout << "\r" << flush;
		}

		//Write to file
		if (sim->getTime() >= next_dump_time) {
			if (write_to_file) {
#ifdef KPSIMULATORAPPS_USE_NETCDF
				buffer->time = sim->getTime();
				sim->getU1(buffer->U[0]->data);
				sim->getU2(buffer->U[1]->data);
				sim->getU3(buffer->U[2]->data);
				file->writeTimeStep(buffer);

#else
				stringstream tmp;
				tmp << output_file << "_" << fixed << setprecision(4) << setw(8) << setfill('0') << sim->getTime() << ".pgm";
				sim->getU1(buffer->data);
				FileManager::writePGMFile(tmp.str().c_str(), buffer);
#endif
			}
			next_dump_time += write_every;
		}
	}
	cout << endl << setw(-1) << setprecision(-1);
	cout << "Simulation of " << sim->getTime() << " seconds took " << sim->getTimeSteps() << " timesteps" << endl;
	cout << "Calculated in " << (end-start) << " (wall clock) seconds." << endl;
	cout << "Average iterations per second = " << sim->getTimeSteps()/(end-start) << " ips." << endl;
	cout << "Average dt = " << sim->getTime() / (float) sim->getTimeSteps() << endl;

	//Write out statistics to file
	if (write_to_file) {
#ifdef KPSIMULATORAPPS_USE_NETCDF/*
		tmp.str("");
		tmp << (end-start);
		file->writeInfoString("wall_time", tmp.str().c_str());

		tmp.str("");
		tmp << sim->getTime();
		file->writeInfoString("simulation_time", tmp.str().c_str());

		tmp.str("");
		tmp << sim->getTimeSteps();
		file->writeInfoString("no_timesteps", tmp.str().c_str());
		file.reset();*/
#endif
	}

	return 0;
}
