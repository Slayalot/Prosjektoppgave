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

//#define SHOW_BOUDARIES

#include "configure.h"

#ifdef KPSIMULATORAPPS_USE_NETCDF

#include <iostream>
#include <string>
#include <iomanip>
#include <omp.h>
#include <vector>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>

#include "KPSimulator.h"
#include "gpu_ptr.hpp"
#include "app/common.hpp"
#include "app/common_visualization.hpp"

#include "FileManager.h"
#include "KPNetCDFFile.h"

// FIXME: These headers are from SINTEF...
#include "PNMImage.hpp"
#include "SWVisualizationContext.h"

class NCWrapper : public SimulatorWrapper {
public:
	NCWrapper(std::string filename) {
		boost::shared_ptr<Field> tmp;

		ifile.reset(new KPNetCDFFile(filename.c_str()));
		ic.reset(new KPInitialConditions(ifile->readInitialConditions(init_B, init_M, ts)));

		KPSIMULATOR_CHECK_CUDA_ERROR(" ");
		for (int i=0; i<3; ++i)
			U_gpu[i].reset(new gpu_ptr_2D<float> (ic->getNx()+2, ic->getNy()+2));
		KPSIMULATOR_CHECK_CUDA_ERROR(" ");

		U_gpu[0]->upload(ts->U[0]->data, 2, 2, ts->U[0]->nx, ts->U[0]->ny);
		U_gpu[1]->upload(ts->U[1]->data, 2, 2, ts->U[1]->nx, ts->U[1]->ny);
		U_gpu[2]->upload(ts->U[2]->data, 2, 2, ts->U[2]->nx, ts->U[2]->ny);

		B_gpu.reset(new gpu_ptr_2D<float> (ic->getNx()+2, ic->getNy()+2));
		tmp.reset(new Field(ic->getNx(), ic->getNy()));
#pragma omp parallel for
		for (int j = 0; j < static_cast<int>(tmp->ny); ++j) {
			for (int i = 0; i < static_cast<int>(tmp->nx); ++i) {
				float sw = ic->getB()[j*ic->getBNx() + i];
				float se = ic->getB()[j*ic->getBNx() + i+1];
				float nw = ic->getB()[(j+1)*ic->getBNx() + i];
				float ne = ic->getB()[(j+1)*ic->getBNx() + i+1];
				tmp->data[j*tmp->nx + i] = 0.25f*(sw+se+nw+ne);
			}
		}
		B_gpu->upload(tmp->data, 2, 2, tmp->nx, tmp->ny);

		iteration = 0;
	}
	virtual void step() {
		ifile->readTimeStepIndex(ts, iteration);
		for (int i=0; i<3; ++i)
			U_gpu[i]->upload(ts->U[i]->data, 2, 2, ts->nx, ts->ny);
		iteration = (iteration+1) % ifile->getNt()+1;
	}
	virtual float getTime() { return ts->time; }
	virtual float getDt() { return 0.0f; }
	virtual int getTimeStep() { return 0; }
	virtual boost::shared_ptr<gpu_ptr_2D<float> > getU1() {return U_gpu[0];}
	virtual boost::shared_ptr<gpu_ptr_2D<float> > getU2() {return U_gpu[1];}
	virtual boost::shared_ptr<gpu_ptr_2D<float> > getU3() {return U_gpu[2];}
	virtual boost::shared_ptr<gpu_ptr_2D<float> > getB() {return B_gpu;}
	
	boost::shared_ptr<KPInitialConditions> getIc() { return ic; }
private:
	boost::shared_ptr<KPInitialConditions> ic;
	boost::shared_ptr<KPNetCDFFile> ifile;
	boost::shared_ptr<TimeStep> ts;
	boost::shared_ptr<Field> init_B;
	boost::shared_ptr<Field> init_M;
	shared_gpu_ptr_2D_array3 U_gpu;
	shared_gpu_ptr_2D B_gpu;
	int iteration;
};


using boost::shared_ptr;
using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;
using std::string;


boost::program_options::variables_map parse_options_nc_vis(int argc, char** argv,
		boost::program_options::options_description& input_options);

shared_ptr<SWVisualizationContext> vis;
shared_ptr<NCWrapper> ncWrapper;




void renderFunc() {
	bool rendered = vis->renderFunc(*ncWrapper.get());
	if (rendered) glutSwapBuffers();
	glutPostRedisplay();
}
void keyboardFunc(unsigned char key, int x, int y) {
	int modifiers = glutGetModifiers();
	vis->keyboardFunc(key, x, y, modifiers);
}
void keyboardUpFunc(unsigned char key, int x, int y) {
	int modifiers = glutGetModifiers();
	vis->keyboardUpFunc(key, x, y, modifiers);
}
void specialFunc(int key, int x, int y) {
	vis->specialFunc(key);
}
void reshapeFunc(int x, int y) { vis->reshapeFunc(x, y); }
void mouseFunc(int b, int s, int x, int y) { vis->mouseFunc(b, s, x, y); }
void passiveMotionFunc(int x, int y) { vis->passiveMotionFunc(x, y); }
void idleFunc() { /*glutPostRedisplay();*/ }


int main(int argc, char** argv) {
	string texture_map_filename;
	string bump_map_filename;
	string normal_map_filename;
	string ncfile_filename;
	shared_ptr<KPInitialConditions> init;
	float scale;

    options_description options("");
    options.add_options()
	("texture_map", value<string>()->default_value(""), "Texture overlay filename")
	("normal_map", value<string>()->default_value(""), "Normal map of terrain filename")
	("bump_map", value<string>()->default_value(""), "Bump map filename")
	("scale", value<float>()->default_value(1.0), "Scale of the visualization");
	variables_map cli_vars = parse_options_nc_vis(argc, argv, options);

	ncfile_filename = cli_vars["ncfile"].as<string>();
	texture_map_filename = cli_vars["texture_map"].as<string>();
	normal_map_filename = cli_vars["normal_map"].as<string>();
	bump_map_filename = cli_vars["bump_map"].as<string>();
	scale = cli_vars["scale"].as<float>();

	//Initialize
    SWVisualizationContext::initGlut(argc, argv);
    ncWrapper.reset(new NCWrapper(ncfile_filename));
	vis = setup_vis(*ncWrapper->getIc().get(),
			texture_map_filename,
			normal_map_filename,
			bump_map_filename,
			scale);

    glutDisplayFunc(renderFunc);
    glutReshapeFunc(reshapeFunc);
    glutKeyboardFunc(keyboardFunc);
    glutSpecialFunc(specialFunc);
    glutKeyboardUpFunc(keyboardUpFunc);
    glutPassiveMotionFunc(passiveMotionFunc);
    glutMouseFunc(mouseFunc);
	glutIdleFunc(idleFunc);

	//Start glut main loop
    glutMainLoop();

	return 0;
}




boost::program_options::variables_map parse_options_nc_vis(int argc, char** argv,
		boost::program_options::options_description& input_options) {
	using namespace boost::program_options;
	using std::cout;
	using std::endl;
	using std::string;
	using std::vector;
	using std::ifstream;

    //Add positional options (so that filenames are parsed as input)
    options_description hidden_options("Hidden options");
    hidden_options.add_options()
	("ncfile", value<string>(), "NetCDF filename");

    positional_options_description pos_opts;
    pos_opts.add("ncfile", 1);

    //Add other command-line options
    options_description options("Commandline options");
    options.add(input_options);
    options.add(hidden_options);
    options.add_options()
    ("help,h", "produce help message")
    ("config,c", value<string>(), "Specify configuration file")
	("texture_map", value<string>(), "Required: Texture overlay filename")
	("normal_map", value<string>(), "Required: Normal map of terrain filename")
    ("bump_map", value<string>(), "Required: Bump map filename");

    //Create commandline parser
    command_line_parser cli_parser(argc, argv);
    cli_parser.positional(pos_opts);
    cli_parser.options(options);

    //Parse, and store in map
    variables_map cli_vars;
    store(cli_parser.run(), cli_vars);
    if (cli_vars.count("config")) {
    	string config_file = cli_vars["config"].as<string>();
    	ifstream ifs(config_file.c_str());
    	store(parse_config_file(ifs, options), cli_vars);
    }
    notify(cli_vars);

    //Now test the variables
    if (cli_vars.count("help")) {
        cout << "Usage: " << argv[0] << " [options] <ncfile>" << endl;
        cout << options << endl;
        exit(1);
    }

    if (!cli_vars.count("ncfile") ||
			!cli_vars.count("texture_map") ||
			!cli_vars.count("normal_map") ||
			!cli_vars.count("bump_map")) {
        cout << "Error getting program options, try --help." << endl;
        exit(-1);
    }

    return cli_vars;
}

#else

#include <iostream>
#include <stdlib.h>

int main(int argc, char** argv) {
	std::cout << "NetCDF support not enabled in makefile" << std::endl;
	exit(-1);
}

#endif //KPSIMULATORAPPS_USE_NETCDF
