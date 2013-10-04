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

#include "configure.h"

#ifdef KPSIMULATORAPPS_USE_NETCDF

#include "KPNetCDFFile.h"
#include "KPSimulator.h"
#include "FileManager.h"

#include <ctime>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <string.h>

using std::cout;
using std::endl;
using std::max;
using std::min;
using std::stringstream;
using std::setfill;
using std::setw;

using boost::shared_ptr;


KPNetCDFFile::KPNetCDFFile(std::string filename) {
	memset(&layout, 0, sizeof(layout));

	if (FileManager::fileExists(filename)) { //Open existing file
		cout << "Opening existing file '" << filename << "' " << endl;
		file.reset(new NcFile(filename.c_str(), NcFile::Write));
		if (!file->is_valid())
			file.reset(new NcFile(filename.c_str(), NcFile::Write, 0, 0, NcFile::Netcdf4));
		if (!file->is_valid()) {
			cout << "Could not open..." << endl;
			exit(-1);
		}
	}
	else { //Create new file
		cout << "Creating file '" << filename << "' " << endl;
#ifdef KPSIMULATORAPPS_USE_NETCDF
		file.reset(new NcFile(filename.c_str(), NcFile::New, 0, 0, NcFile::Netcdf4));
#else
		file.reset(new NcFile(filename.c_str(), NcFile::New));
#endif
		if (!file->is_valid()) {
			cout << "Could not create '" << filename << "'." << endl;
			cout << "Check that it does not exist, or that your disk is full." << endl;
			exit(-1);
		}
	}
}

KPNetCDFFile::~KPNetCDFFile() {
	file->sync();
	if (!file->close()) {
		cout << "Error: Couldn't close NetCDF file!" << endl;
	}	
	/** 
	  * This generates a segfault on MSVC:
	  * FIX: Explicit Call to Virtual Destructor Corrupts Stack
	  * http://support.microsoft.com/kb/128805
	  * 
	  * The fix is to use the following in the netcdf header files: 
	  * (simply use the preprocessor to remove virtual)
	  * (also when building the netcdf dlls)
	  *
	  * #ifndef WIN32
	  * virtual 
	  * #endif
	  * ~NcFile();
	  *
	  */
	file.reset();
}

void KPNetCDFFile::writeDims(const KPInitialConditions& init) {
	float* tmp = new float[max(init.getBNx(), init.getBNy())];

	//Create dimensions
	layout.dims.i = file->add_dim("I", init.getBNx());
	layout.dims.j = file->add_dim("J", init.getBNy());
	layout.dims.x = file->add_dim("X", init.getNx());
	layout.dims.y = file->add_dim("Y", init.getNy());
	layout.dims.t = file->add_dim("T");

	//Create indexing variables
	layout.vars.i = file->add_var("I", ncFloat, layout.dims.i);
	layout.vars.j = file->add_var("J", ncFloat, layout.dims.j);
	layout.vars.i->add_att("description", "Longitudal coordinate for values given at grid cell intersections");
	layout.vars.j->add_att("description", "Latitudal coordinate for values given at grid cell intersections");

	layout.vars.x = file->add_var("X", ncFloat, layout.dims.x);
	layout.vars.y = file->add_var("Y", ncFloat, layout.dims.y);
	layout.vars.x->add_att("description", "Longitudal coordinate for values given at grid cell centers");
	layout.vars.y->add_att("description", "Latitudal coordinate for values given at grid cell centers");

	layout.vars.t = file->add_var("T", ncFloat, layout.dims.t);
	layout.vars.t->add_att("description", "Time");

	//Write contents of spatial variables
	for (unsigned int i=0; i<init.getBNx(); ++i)
		tmp[i] = i * init.getDx();
	layout.vars.i->put(tmp, init.getBNx());

	for (unsigned int i=0; i<init.getBNy(); ++i)
		tmp[i] = i * init.getDy();
	layout.vars.j->put(tmp, init.getBNy());

	for (unsigned int i=0; i<init.getNx(); ++i)
		tmp[i] = (i+0.5f) * init.getDx();
	layout.vars.x->put(tmp, init.getNx());

	for (unsigned int i=0; i<init.getNy(); ++i)
		tmp[i] = (i+0.5f) * init.getDy();
	layout.vars.y->put(tmp, init.getNy());

	delete [] tmp;
	file->sync();
}

void KPNetCDFFile::readDims() {
	//Get dimensions
	layout.dims.i = file->get_dim("I");
	layout.dims.j = file->get_dim("J");
	layout.dims.x = file->get_dim("X");
	layout.dims.y = file->get_dim("Y");
	layout.dims.t = file->get_dim("T");

	//Get indexing variables
	layout.vars.i = file->get_var("I");
	layout.vars.j = file->get_var("J");

	layout.vars.x = file->get_var("X");
	layout.vars.y = file->get_var("Y");

	layout.vars.t = file->get_var("T");
}

void KPNetCDFFile::writeVars(const KPInitialConditions& init) {
	//Create initial condidion variables
	layout.vars.init_B = file->add_var("init_B", ncFloat, layout.dims.j, layout.dims.i);
	layout.vars.init_B->add_att("description", "Initial conditions for bathymetry");

	if (init.getM().spatially_varying) {
		layout.vars.init_M = file->add_var("init_M", ncFloat, layout.dims.y, layout.dims.x);
		layout.vars.init_M->add_att("description", "Initial conditions for Manning coefficient");
	}

	//Create the timestep variables
	layout.vars.U1 = file->add_var("U1", ncFloat, layout.dims.t, layout.dims.y, layout.dims.x);
	layout.vars.U2 = file->add_var("U2", ncFloat, layout.dims.t, layout.dims.y, layout.dims.x);
	layout.vars.U3 = file->add_var("U3", ncFloat, layout.dims.t, layout.dims.y, layout.dims.x);

	layout.vars.U1->add_att("description", "Water elevation");
	layout.vars.U2->add_att("description", "Longitudal water discharge");
	layout.vars.U3->add_att("description", "Latitudal water discharge");


#ifdef KPSIMULATORAPPS_USE_NETCDF_COMPRESSION
	nc_def_var_deflate(file->id(), layout.vars.init_B->id(), 1, 1, 2);
	if (init.getM().spatially_varying) {
		nc_def_var_deflate(file->id(), layout.vars.init_M->id(), 1, 1, 2);
	}
	nc_def_var_deflate(file->id(), layout.vars.U1->id(), 1, 1, 2);
	nc_def_var_deflate(file->id(), layout.vars.U2->id(), 1, 1, 2);
	nc_def_var_deflate(file->id(), layout.vars.U3->id(), 1, 1, 2);
#endif
	file->sync();
}

void KPNetCDFFile::readVars() {
	//Create initial condidion variables
	layout.vars.init_B = file->get_var("init_B");
	for (int i=0; i<file->num_vars(); ++i) {
		const std::string token("init_M");
		const char* var_name = file->get_var(i)->name();
		if (token.compare(var_name) == 0) {
			layout.vars.init_M = file->get_var("init_M");
		}
	}

	//Create the timestep variables
	layout.vars.U1 = file->get_var("U1");
	layout.vars.U2 = file->get_var("U2");
	layout.vars.U3 = file->get_var("U3");
}


void KPNetCDFFile::writeAtts(const KPInitialConditions& init) {
	stringstream tmp;

	tmp.str("");
	tmp << "KPSimulator version " << KPSimulator::versionString();
	file->add_att("created_by", tmp.str().c_str());

	tmp.str("");
	time_t foo = time(NULL);
	tm* t = localtime(&foo);
	tmp << setfill('0');
	tmp << (1900+t->tm_year) << "-" << setw(2) << t->tm_mon << "-" << setw(2) << t->tm_mday;
	tmp << " " << setw(2) << t->tm_hour << ":" << setw(2) << t->tm_min << ":" << setw(2) << t->tm_sec;
	file->add_att("created_on", tmp.str().c_str());

	file->add_att("created_with", "coffee and salad");

	file->sync();
}


void KPNetCDFFile::readAtts() {
	cout << "created_by   = " << file->get_att("created_by")->as_string(0) << endl;
	cout << "created_on   = " << file->get_att("created_on")->as_string(0) << endl;
	cout << "created_with = " << file->get_att("created_with")->as_string(0) << endl;
}


void KPNetCDFFile::writeInitialConditions(const KPInitialConditions& init) {
	writeDims(init);
	writeVars(init);
	writeAtts(init);
	nt = 0;
	time_offset = 0;

	//Write all attributes that describe the simulation
	file->add_att("nx", (int) init.getNx());
	file->add_att("ny", (int) init.getNy());
	file->add_att("dx", init.getDx());
	file->add_att("dy", init.getDy());
	file->add_att("g", init.getG());
	file->add_att("dt_scale", init.getDtScale());
	file->add_att("time_integrator", init.getTimeIntegrator());
	file->add_att("bc_north", init.getNorthBC().getType());
	file->add_att("bc_south", init.getSouthBC().getType());
	file->add_att("bc_east", init.getEastBC().getType());
	file->add_att("bc_west", init.getWestBC().getType());
	//file->add_att("bc_north_arg", init.getBC().getNorthArg());
	//file->add_att("bc_south_arg", init.getBC().getSouthArg());
	//file->add_att("bc_east_arg", init.getBC().getEastArg());
	//file->add_att("bc_west_arg", init.getBC().getWestArg());

	//Write initial conditions
	layout.vars.init_B->put(init.getB(), init.getBNy(), init.getBNx());
	if (init.getM().spatially_varying == false) {
		file->add_att("n", init.getM().n[0]);
	}
	else {
		layout.vars.init_M->put(init.getM().n, init.getUNy(), init.getUNx());
	}

	file->sync();
}


KPInitialConditions KPNetCDFFile::readInitialConditions(shared_ptr<Field>& init_B,
		boost::shared_ptr<Field>& init_M,
		boost::shared_ptr<TimeStep>& ts) {
	KPInitialConditions::KPParameters params;
	readDims();
	readVars();
	readAtts();
	nt = layout.dims.t->size() - 1;
	layout.vars.t->set_cur(0l);
	layout.vars.t->get(&time_offset, 1);

	//Read all attributes that describe the simulation
	params.nx = file->get_att("nx")->as_int(0);
	params.ny = file->get_att("ny")->as_int(0);
	params.dx = file->get_att("dx")->as_float(0);
	params.dy = file->get_att("dy")->as_float(0);
	params.g = file->get_att("g")->as_float(0);
	params.dt_scale = file->get_att("dt_scale")->as_float(0);
	params.ode = (KPInitialConditions::TIME_INTEGRATOR) file->get_att("time_integrator")->as_int(0);

	KPBoundaryCondition::TYPE bc_north = (KPBoundaryCondition::TYPE) file->get_att("bc_north")->as_int(0);
	KPBoundaryCondition::TYPE bc_south = (KPBoundaryCondition::TYPE) file->get_att("bc_south")->as_int(0);
	KPBoundaryCondition::TYPE bc_east = (KPBoundaryCondition::TYPE) file->get_att("bc_east")->as_int(0);
	KPBoundaryCondition::TYPE bc_west = (KPBoundaryCondition::TYPE) file->get_att("bc_west")->as_int(0);
	float bc_north_arg = 0.; //file->get_att("bc_north_arg")->as_float(0);
	float bc_south_arg = 0.; //file->get_att("bc_south_arg")->as_float(0);
	float bc_east_arg = 0.; //file->get_att("bc_east_arg")->as_float(0);
	float bc_west_arg = 0.; //file->get_att("bc_west_arg")->as_float(0);
	params.bc_north = KPBoundaryCondition(bc_north, bc_north_arg);
	params.bc_south = KPBoundaryCondition(bc_south, bc_south_arg);
	params.bc_east = KPBoundaryCondition(bc_east, bc_east_arg);
	params.bc_west = KPBoundaryCondition(bc_west, bc_west_arg);
	

	init_B.reset(new Field(params.nx+1, params.ny+1));
	params.B = init_B->data;
	layout.vars.init_B->get(init_B->data, init_B->ny, init_B->nx);

	if (layout.vars.init_M == NULL) {
		init_M.reset(new Field(1, 1));
		params.M.n = init_M->data;
		params.M.spatially_varying = false;
		init_M->data[0] = file->get_att("n")->as_float(0);
	}
	else {
		init_M.reset(new Field(params.nx, params.ny));
		params.M.n = init_M->data;
		params.M.spatially_varying = true;
		layout.vars.init_M->get(init_M->data, init_M->ny, init_M->nx);
	}

	ts.reset(new TimeStep(params.nx, params.ny));
	params.U1 = ts->U[0]->data;
	layout.vars.U1->get(ts->U[0]->data, 1, ts->U[0]->ny, ts->U[0]->nx);
	params.U2 = ts->U[1]->data;
	layout.vars.U2->get(ts->U[1]->data, 1, ts->U[1]->ny, ts->U[1]->nx);
	params.U3 = ts->U[2]->data;
	layout.vars.U3->get(ts->U[2]->data, 1, ts->U[2]->ny, ts->U[2]->nx);

	layout.vars.U1->set_cur(0l);
	layout.vars.U2->set_cur(0l);
	layout.vars.U3->set_cur(0l);

	//Create initial conditions variable
	return KPInitialConditions(params);
}


void KPNetCDFFile::writeTimeStep(boost::shared_ptr<TimeStep> ts, int index) {
	t_index = index;
	if (t_index < 0) t_index = nt;

	float t = ts->time + time_offset;
	layout.vars.t->set_cur(t_index);
	layout.vars.t->put(&t, 1);

	layout.vars.U1->set_cur(t_index, 0, 0);
	layout.vars.U1->put(ts->U[0]->data, 1, ts->ny, ts->nx);

	layout.vars.U2->set_cur(t_index, 0, 0);
	layout.vars.U2->put(ts->U[1]->data, 1, ts->ny, ts->nx);

	layout.vars.U3->set_cur(t_index, 0, 0);
	layout.vars.U3->put(ts->U[2]->data, 1, ts->ny, ts->nx);

	file->sync();
	nt++;
}


void KPNetCDFFile::readTimeStepIndex(boost::shared_ptr<TimeStep> ts, int index) {
	if (index >= 0) t_index = index;
	if (t_index >= nt) t_index = nt;

	layout.vars.t->set_cur(t_index);
	layout.vars.t->get(&ts->time, 1);

	layout.vars.U1->set_cur(t_index, 0, 0);
	layout.vars.U1->get(ts->U[0]->data, 1, ts->ny, ts->nx);

	layout.vars.U2->set_cur(t_index, 0, 0);
	layout.vars.U2->get(ts->U[1]->data, 1, ts->ny, ts->nx);

	layout.vars.U3->set_cur(t_index, 0, 0);
	layout.vars.U3->get(ts->U[2]->data, 1, ts->ny, ts->nx);

	float dx = file->get_att("dx")->as_float(0);
	float dy = file->get_att("dy")->as_float(0);

	for (int i=0; i<3; ++i) {
		ts->U[i]->dx = dx;
		ts->U[i]->dy = dy;
	}

	file->sync();
	t_index++;
}

void KPNetCDFFile::readTimeStepTime(boost::shared_ptr<TimeStep> ts, float time) {
	int index = 0;
	float time_tmp;
	float time_diff_min;

	layout.vars.t->set_cur(index);
	layout.vars.t->get(&time_tmp, 1);

	time_diff_min = std::abs(time-time_tmp);
	for (int i=1; i<nt; ++i) {
		layout.vars.t->set_cur(i);
		layout.vars.t->get(&time_tmp, 1);

		float time_diff_new = std::abs(time-time_tmp);
		if (time_diff_new < time_diff_min) {
			time_diff_min = time_diff_new;
			index = i;
		}
	}

	return readTimeStepIndex(ts, index);
}

unsigned int KPNetCDFFile::getNt() {
	return nt;
}

#endif //KPSIMULATORAPPS_USE_NETCDF
