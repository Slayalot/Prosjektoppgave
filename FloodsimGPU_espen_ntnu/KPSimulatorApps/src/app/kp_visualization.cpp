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

#include "KPSimulator.h"
#include "app/common.hpp"
#include "app/common_visualization.hpp"

#include "CommandLineManager.h"
#include "InitialConditionsManager.h"
#include "FileManager.h"

#include "PNMImage.hpp"
#include "SWVisualizationContext.h"
#include "KPSimWrapper.h"
#include "ScreenshotCreator.h"

using boost::shared_ptr;
using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using std::string;

shared_ptr<SWVisualizationContext> vis;
shared_ptr<KPSimWrapper> simWrapper;

#define DEBUG_TEX
std::vector<float> h;
unsigned int width;
unsigned int height;
void renderFunc() {
	bool rendered = vis->renderFunc(*simWrapper.get());
	if (rendered) {
#define DEBUG_TEX
#ifdef DEBUG_TEX
		simWrapper->drawDebugTexture();
#endif
		glutSwapBuffers();
	}
	glutPostRedisplay();
    
    h.resize(width*height);
    simWrapper->getU1()->download(&h[0], 2, 2, width, height);
    double h_sum = 0.0;
    for (unsigned int i=0; i<width*height; ++i) {
        h_sum += h[i];
}
    std::cout << "H sum = " << h_sum << "\r";
    std::cout.flush();
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
	string recorder_filename;

	float vertical_scale;

    options_description options("");
    options.add_options()
				("texture_map", value<string>()->default_value(""), "Texture overlay filename")
				("normal_map", value<string>()->default_value(""), "Normal map of terrain filename")
			    ("bump_map", value<string>()->default_value(""), "Bump map filename")
			    ("vertical_scale", value<float>()->default_value(1.0), "Vertical scale of the visualization")
				("recorder", value<string>()->default_value("recorder.dat"), "Recorder filename (for creating movies)");
	CommandLineManager cli(options);
	variables_map cli_vars = cli.getVariablesMap(argc, argv);

	InitialConditionsManager ic_manager(cli_vars);
	KPInitialConditions init = ic_manager.getIC();

	texture_map_filename = cli_vars["texture_map"].as<string>();
	normal_map_filename = cli_vars["normal_map"].as<string>();
	bump_map_filename = cli_vars["bump_map"].as<string>();
	recorder_filename = cli_vars["recorder"].as<string>();
	vertical_scale = cli_vars["vertical_scale"].as<float>();

	//Initialize
	SWVisualizationContext::initGlut(argc, argv);
	vis = setup_vis(init,
			texture_map_filename,
			normal_map_filename,
			bump_map_filename, vertical_scale);
	simWrapper.reset(new KPSimWrapper(init));

    width = init.getUNx();
    height = init.getUNy();

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
