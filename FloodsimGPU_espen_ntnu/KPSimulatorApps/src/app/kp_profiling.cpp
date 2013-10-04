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

#include <boost/shared_ptr.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/options_description.hpp>

#include "CommandLineManager.h"
#include "InitialConditionsManager.h"
#include "datatypes.h"
#include "KPSimulator.h"
#include "app/common.hpp"


using boost::shared_ptr;
using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;
using std::cout;
using std::endl;
using std::fixed;
using std::setprecision;


int main(int argc, char** argv) {
	float total_time, write_every, next_dump_time=0.0;
	shared_ptr<KPSimulator> sim;
	shared_ptr<Field> B, U1, U2, U3, M;
	double start=0.0, end=0.0;
	double real_start=0.0, real_end=0.0;
	bool use_real_time = false;

	//Parse command line options and generate initial conditions from it
    options_description options("Profiling options");
    options.add_options()
				("total_time", value<float>(), "Required (total_time or real_time): Total simulation time")
				("real_time", value<float>(), "Required (total_time or real_time): Total (wall clock) time")
				("write_every", value<float>()->default_value(-1.0f), "Write every t seconds to screen");
	CommandLineManager cli(options);
	variables_map cli_vars = cli.getVariablesMap(argc, argv);
	InitialConditionsManager ic_manager(cli_vars);
	KPInitialConditions init = ic_manager.getIC();
	if (cli_vars.count("total_time") && cli_vars.count("write_every")) {
		total_time = cli_vars["total_time"].as<float>();
	} else if (cli_vars.count("real_time") && cli_vars.count("write_every")) {
		use_real_time = true;
		total_time = cli_vars["real_time"].as<float>();
	} else {
		cout << "Error getting program options, try --help." << endl;
		exit(-1);
	}
	write_every = cli_vars["write_every"].as<float>();

	sim.reset(new KPSimulator(init));
	cout << sim.get() << endl;

	cout << "simulation_timestep, simulation_time, wall_clock" << endl;
	cout << fixed;
	cout << setprecision(20);

	if (write_every < 0.0f) {
		next_dump_time = total_time;
		write_every = total_time;
	}

	if(use_real_time) {
		start = getCurrentTime();
	} else {
		real_start = getCurrentTime();
		start = sim->getTime();
	}
	end = start;
	while (next_dump_time <= total_time) {
		while (end-start < next_dump_time) {
			sim->step();
			if (use_real_time) {
				end = getCurrentTime();
			} else {
				real_end = getCurrentTime();
				end = sim->getTime();
			}
		}

		if (use_real_time) {
			cout << sim->getTimeSteps() << ", " << sim->getTime() << ", " << end - start << endl;
		} else {
			cout << sim->getTimeSteps() << ", " << sim->getTime() << ", " << real_end - real_start << endl;
		}
		next_dump_time += write_every;
	}

	return 0;
}
