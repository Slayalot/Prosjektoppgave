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

#include "SWVisualizationContext.h"

#include <vector>
#include <iostream>
#include <iomanip>
#include <cctype>
#include <ctime>
#include <limits>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "util.hpp"
#include "gpu_ptr.hpp"
#include "KPInitialConditions.h"

//#define WIREFRAME

using namespace std;
using namespace util;


#ifdef _WIN32
#include <sys/timeb.h>
#include <time.h>
#include <windows.h>
#else
#include <sys/time.h>
#include <unistd.h>
#endif

inline double getCurrentTime() {
#if defined(_WIN32)
	LARGE_INTEGER f;
	LARGE_INTEGER t;
	QueryPerformanceFrequency(&f);
	QueryPerformanceCounter(&t);
	return t.QuadPart/(double) f.QuadPart;
#else
	struct timeval tv;
	struct timezone tz;
	gettimeofday(&tv, &tz);
	return tv.tv_sec+tv.tv_usec*1e-6;
#endif
}

SWVisualizationContext::SWVisualizationContext(KPInitialConditions& init, float vertical_scale_) {
	this->init(init.getNx(), init.getNy(), init.getDx(), init.getDy(), vertical_scale_);
}

boost::shared_ptr<CameraTransform> SWVisualizationContext::getCameraTransform() {
	return ct;
}
boost::shared_ptr<Renderer> SWVisualizationContext::getRenderer() {
	return rd;
}

const SWVisualizationContext::WATER_RENDER_TYPE& SWVisualizationContext::getWaterRenderType() {
	return water_render_mode;
}

void SWVisualizationContext::init(size_t width, size_t height, float dx, float dy, float dz) {
	this->width = width;
	this->height = height;
	this->dx = dx;
	this->dy = dy;
	this->dz = dz;

	this->screen_width = 1280;
	this->screen_height = 720;
	this->screen_center_x = screen_width >> 1;
	this->screen_center_y = screen_height >> 1;

	this->u_data_changed = true;
	this->b_data_changed = true;
	this->playback = false;
	this->playback_position = 0;
	this->capture_mouse = false;
	this->sim_speed = 0;
	this->simulate = false;
	this->screenshots = false;
	this->record = false;
	this->wireframe = false;
	this->skybox = true;
	this->rotate = false;
	this->rotate_degree = 0.0f;
	this->ips_avg = 0;
	this->fps_avg = 0;
	this->speedup_avg = 0;
	this->dt_avg = 0;
	this->iters = 0;
	this->fps_rate = 30;
	this->recorder_filename = "sw_recorder.dat";

	this->x_aspect = 1;
	this->y_aspect = 1;
	this->zoom = 1;

	this->next_passivemotion_time = getCurrentTime();
	this->last_render_time = next_passivemotion_time;
	this->next_render_time = last_render_time + 1.0f/fps_rate;
	this->next_simulation_time = 0;

	readRecorder();

	this->water_render_mode = WATER_FRESNEL;
	for(int i=0; i<256; ++i) keyDown[i] = false;

	initGLState();

	float scale = 1.5f*max(dx*width,dy*height);
	this->camera_speed = 1e-1f*scale;
	this->fz0 = 1e-3f*scale;
	this->fz1 = scale;
	
	this->step = false;

	CameraState camera(0.0f,  -0.5f*min(dx*width,dy*height), 0.0f, 135.0f, 25.0f);
	ct.reset(new CameraTransform(camera));
	rd.reset(new Renderer(width, height, dx, dy, dz));
	sc.reset(new ScreenshotCreator("sw_vis"));
}

void SWVisualizationContext::setProjectionMatrix() {
	//Set up projection matrix
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(75*zoom, screen_width/(float) screen_height, fz0, fz1);
    glViewport(0, 0, screen_width, screen_height);
    glMatrixMode(GL_MODELVIEW);
}


void SWVisualizationContext::initGLState() {
	setProjectionMatrix();

    //Set up modelview matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glClearColor(0.0, 0.0, 0.0, 0.0);

    glEnable(GL_TEXTURE_2D);
    glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	CHECK_GL_ERRORS();
}


void SWVisualizationContext::initGlut(int argc, char** argv) {
#ifdef FREEGLUT
	glutInitContextVersion(2, 1);
#endif
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(1280, 720);
    glutCreateWindow("FloodSimGPU - (c) 2010, 2011 SINTEF");

#ifdef FREEGLUT
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
#endif

    glewInit();
    CHECK_GL_ERRORS();
}



template <typename T>
inline T* address2D(T* base, unsigned int pitch, unsigned int x, unsigned int y) {
	return (T*) ((char*) base+y*pitch) + x;
}

bool SWVisualizationContext::renderFunc(SimulatorWrapper& sim) {
	const float weight = 0.1f;

	if (playback) {
		playRecord();
		while (sim.getTime() < sim_time) sim.step();
		u_data_changed = true;

		updateData(sim);
		renderScene(0);
		return true;
	}
	else {
		runSimulation(sim);
		double now = getCurrentTime();
		if (now >= next_render_time) {
			stringstream stats;
			float elapsed = (float) (now - last_render_time);
			ips_avg = (1.0f-weight)*ips_avg + weight*(iters/elapsed);
			fps_avg = (1.0f-weight)*fps_avg + weight*(1.0f/elapsed);
			dt_avg = (1.0f-weight)*dt_avg + weight*sim.getDt();
			speedup_avg = (1.0f-weight)*speedup_avg + weight*(ips_avg*dt_avg);

			stats.str("");
			stats << "View=";
			switch(water_render_mode) {
			case WATER_FRESNEL:  stats << "photorealistic";  break;
			case WATER_DEPTH:    stats << "depth";    break;
			case WATER_VELOCITY: stats << "velocity"; break;
			}
			stats << std::endl;

			stats << "FPS=" << fixed << setprecision(0) << fps_avg << " (limit=" << fps_rate << ")";
			stats << ", Speed=" << fixed << setprecision(0) << speedup_avg << " (limit=" << sim_speed << ")";
			stats << std::endl;

			stats << "t=" << printTime(sim.getTime());
			stats << ", ts=" << sim.getTimeStep();
			stats << ", Dt=" << scientific << setprecision(2) << dt_avg << " s";
			stats << ", IPS=" << fixed << setprecision(0) << ips_avg;
			osd = stats.str();

			updateData(sim);
			renderScene(elapsed);

			last_render_time = now;
			next_render_time = now + 1.0/fps_rate;
			if (sim.getTime() >= next_simulation_time) {
				next_simulation_time += sim_speed/fps_rate;
			}
			iters = 0;
			if (record) addRecord(sim.getTime());
			if (rotate) rotate_degree = (rotate_degree + 10*elapsed);
			if (rotate_degree > 360) rotate_degree -= 360;

			return true;
		}
	}

	return false;
}

void SWVisualizationContext::runSimulation(SimulatorWrapper& sim) {
	if(step) {
		sim.step();
		step = false;
		u_data_changed = true;
	}
	else if (simulate) {
		if (sim.getTime() < next_simulation_time) { //Prioritize simulation over fps count...
			sim.step();
			iters++;
		}
		if (sim_speed > 0) {
			while (sim.getTime() < next_simulation_time && getCurrentTime() < next_render_time) {
				sim.step();
				iters++;
			}
		}
		else {
			while (getCurrentTime() < next_render_time) {
				sim.step();
				iters++;
			}
			next_simulation_time = sim.getTime()+1e-6;
		}
		u_data_changed = true;
	}
}

void SWVisualizationContext::updateData(SimulatorWrapper& sim) {
	if (u_data_changed) {
		updateUData(sim.getU1(), sim.getU2(), sim.getU3());
		u_data_changed = false;
	}
	if (b_data_changed) {
		updateBData(sim.getB());
		b_data_changed = false;
	}
}

void SWVisualizationContext::renderScene(float elapsed) {
	float delta = camera_speed * elapsed;

	glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

	ct->rotate(keyDown['j'], keyDown['l'], keyDown['i'], keyDown['k'], delta);
	ct->translate(keyDown['w'], keyDown['s'], keyDown['a'], keyDown['d'], keyDown['q'], keyDown['e'], delta);

	if (wireframe) {
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}
	if (skybox) rd->renderSkybox();
	else glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glTranslatef(0.5f*width*dx, 0, 0.5f*height*dy);
	glRotatef(rotate_degree, 0.0f, 1.0f, 0.0f);
	glTranslatef(-0.5f*width*dx, 0.0f, -0.5f*height*dy);

	rd->renderLandscape();
	switch(water_render_mode) {
	case WATER_FRESNEL:  rd->renderWaterFresnel();  break;
	case WATER_DEPTH:    rd->renderWaterDepth();    break;
	case WATER_VELOCITY: rd->renderWaterVelocity(); break;
	}

	if (wireframe) {
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}

	rd->renderString(osd);

	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	if (screenshots) sc->snap();
}

void SWVisualizationContext::addRecord(float sim_time) {
	SWVisRecord record;
	record.camera = ct->getCameraState();
	record.wireframe = wireframe;
	record.water_render_mode = water_render_mode;
	record.rotate_degree = rotate_degree;
	record.sim_time = sim_time;
	record.osd = osd;
	recorder.push_back(record);
}

void SWVisualizationContext::playRecord() {
	capture_mouse = false;
	record = false;

	if (playback_position >= (int) recorder.size()) {
		playback = false;
		screenshots = false;
		return;
	}

	ct->setCameraState(recorder.at(playback_position).camera);
	wireframe = recorder.at(playback_position).wireframe;
	water_render_mode = recorder.at(playback_position).water_render_mode;
	rotate_degree = recorder.at(playback_position).rotate_degree;
	sim_time = recorder.at(playback_position).sim_time;
	osd = recorder.at(playback_position).osd;

	playback_position = playback_position + 1;
}

/**
  * Simply writes the "tape" record to disk
  */
void SWVisualizationContext::saveRecorder() {
	ofstream file;

	file.open(recorder_filename.c_str(), ios::in);

	if (file) {
		cout << "File " << recorder_filename << " already exists. Aborting write to file..." << endl;
		file.close();
		return;
	}

	file.close();
	file.open(recorder_filename.c_str(), ios::out);
	if (!file)
		throw "Unable to open file";

	//Then the records sequentially...
	for (unsigned int i=0; i<recorder.size(); ++i) {
		file << recorder[i].camera.x << " ";
		file << recorder[i].camera.y << " ";
		file << recorder[i].camera.z << " ";
		file << recorder[i].camera.u << " ";
		file << recorder[i].camera.v << " ";
		file << recorder[i].rotate_degree << " ";
		file << recorder[i].wireframe << " ";
		file << recorder[i].water_render_mode << " ";
		file << recorder[i].sim_time << endl;
		file << "OSD_BEGIN" << endl;
		file << recorder[i].osd << endl;
		file << "OSD_END" << endl;
	}

	//and close file
	file.close();
}

/** 
  * Simply reads the "tape" from disk
  */
void SWVisualizationContext::readRecorder() {
	ifstream file;
	string tmp;
	int render_mode;
	file.open(recorder_filename.c_str(), ios::in);
	if (!file) {
		cout << "Unable to open " << recorder_filename << endl;
		file.close();
		return;
	}

	recorder.clear();

	//Then the records sequentially...
	while (getline(file, tmp)) {
		SWVisRecord record;
		istringstream iss(tmp);
		iss >> record.camera.x;
		iss >> record.camera.y;
		iss >> record.camera.z;
		iss >> record.camera.u;
		iss >> record.camera.v;
		iss >> record.rotate_degree;
		iss >> record.wireframe;
		iss >> render_mode;
		record.water_render_mode = (SWVisualizationContext::WATER_RENDER_TYPE) render_mode;
		iss >> record.sim_time;

		getline(file, tmp);
		if (tmp.compare("OSD_BEGIN")) {
			cout << "Error reading " << recorder_filename << ": expecting OSD_BEGIN" << endl;
			return;
		}

		getline(file, tmp);
		record.osd.append(tmp);
		getline(file, tmp);
		while (tmp.compare("OSD_END")) {
			record.osd.append("\n");
			record.osd.append(tmp);
			getline(file, tmp);
		}

		recorder.push_back(record);
	}

	//and close file
	file.close();
}

void SWVisualizationContext::reshapeFunc(int x, int y) {
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
	
    if ( x <= y ) {
		y_aspect = y / (GLfloat) x;
		x_aspect = 1.0f;
    } 
	else {
		y_aspect = 1.0f;
		x_aspect = x / (GLfloat) y;
    }


    screen_width = x;
    screen_height = y;
	screen_center_x = x >> 1;
	screen_center_y = y >> 1;
	if (capture_mouse) glutWarpPointer(screen_center_x, screen_center_y);

    setProjectionMatrix();
    glMatrixMode(GL_MODELVIEW);
}

void SWVisualizationContext::keyboardFunc(unsigned char key, int x, int y, int modifiers) {
	modifiers = glutGetModifiers();
	switch (key) {
		case '1':
			water_render_mode = WATER_FRESNEL;
			break;
		case '2':
			water_render_mode = WATER_DEPTH;
			break;
		case '3':
			water_render_mode = WATER_VELOCITY;
			u_data_changed = true;
			break;
		case 32:
			simulate = !simulate;
			cout << "Simulator is " << (simulate?"running":"paused") << endl;
			break;
		case 19: //ctrl+s
			cout << "Saving recorder to " << recorder_filename << endl;
			saveRecorder();
			break;
		case 15: //ctrl+o
			cout << "Reading recorder from " << recorder_filename << endl;
			readRecorder();
			break;
		case 17: //CTRL+q
		case 27: //(ESC)
#ifdef FREEGLUT
			glutLeaveMainLoop();
#else
			exit(0);
#endif
		case '+':
			sim_speed++;
			break;
		case '-':
			sim_speed = std::max(sim_speed-1, 0.0f);
			break;
	}

	keyDown[(int)key] = true;
}

void SWVisualizationContext::keyboardUpFunc(unsigned char key, int x, int y, int modifiers) {
	keyDown[(int)key] = false;
}


void SWVisualizationContext::specialFunc(unsigned char key) {
	switch (key) {
	case GLUT_KEY_F1:
		screenshots = !screenshots;
		cout << "Screenshots " << (screenshots?"on":"off") << endl;
		break;
	case GLUT_KEY_F2:
		playback = !playback;
		playback_position = 0;
		cout << "Playing path " << (playback?"on":"off") << endl;
		break;
	case GLUT_KEY_F3:
		record = !record;
		cout << "Recording path " << (record?"on":"off") << endl;
		break;
	case GLUT_KEY_F9:
		wireframe = !wireframe;
		cout << "Wireframe " << (wireframe?"on":"off") << endl;
		break;
	case GLUT_KEY_F10:
		skybox = !skybox;
		cout << "Skybox " << (skybox?"on":"off") << endl;
		break;
	case GLUT_KEY_F11:
		rotate = !rotate;
		cout << "Rotate " << (rotate?"on":"off") << endl;
		break;
	case GLUT_KEY_PAGE_UP:
		camera_speed *= 1.3f;
		cout << "Camera speed is " << camera_speed << endl;
		break;
	case GLUT_KEY_PAGE_DOWN:
		camera_speed /= 1.3f;
		cout << "Camera speed is " << camera_speed << endl;
		break;
	case GLUT_KEY_F5: //step
		step = true;
		break;
	}
}

void SWVisualizationContext::mouseFunc(int b, int s, int x, int y) {
    //Left mouse down
    if (s==GLUT_DOWN && b == GLUT_LEFT_BUTTON) {
    	capture_mouse = !capture_mouse;

		if (capture_mouse) {
			glutSetCursor(GLUT_CURSOR_NONE);
			glutWarpPointer(screen_center_x, screen_center_y);
		}
		else {
			glutSetCursor(GLUT_CURSOR_INHERIT);
		}
    }
}

void SWVisualizationContext::passiveMotionFunc(int x, int y) {
	if (getCurrentTime() > next_passivemotion_time) {
		float delta = 1.0f/50.0f;
		next_passivemotion_time += 1.0f/100.0f;

		if (capture_mouse) {
			if (x != screen_center_x && y != screen_center_y) {
				glMatrixMode(GL_MODELVIEW);
				glPushMatrix();
				ct->rotate(delta*(x - (float) screen_center_x), delta*(y - (float) screen_center_y));
				glPopMatrix();

				glutWarpPointer(screen_center_x, screen_center_y);
			}
		}
	}
}



void SWVisualizationContext::memcpyCudaToOgl(GLuint dst, size_t w_offset, size_t h_offset, float* src, size_t pitch, size_t width_in_bytes, size_t height) {
	cudaGraphicsResource* res[1];
  	cudaArray* array;

	KPSIMULATOR_CHECK_CUDA(cudaGraphicsGLRegisterImage(&res[0], dst, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));
	KPSIMULATOR_CHECK_CUDA(cudaGraphicsMapResources(1, res, 0));
	KPSIMULATOR_CHECK_CUDA(cudaGraphicsSubResourceGetMappedArray(&array, res[0], 0, 0));
	KPSIMULATOR_CHECK_CUDA(cudaMemcpy2DToArray(array, w_offset, h_offset, src, pitch, width_in_bytes, height, cudaMemcpyDeviceToDevice));
	KPSIMULATOR_CHECK_CUDA(cudaGraphicsUnmapResources(1, res, 0));
	KPSIMULATOR_CHECK_CUDA(cudaGraphicsUnregisterResource(res[0]));
	CHECK_GL_ERRORS();
}

void SWVisualizationContext::updateBData(boost::shared_ptr<gpu_ptr_2D<float> > B) {
	GLuint Btex = rd->getBottomTex();

	memcpyCudaToOgl(Btex, 0, 0,
			address2D(B->getRawPtr().ptr, B->getRawPtr().pitch, 2, 2),
			B->getRawPtr().pitch,
			(width)*sizeof(float), height);
}

void SWVisualizationContext::updateUData(boost::shared_ptr<gpu_ptr_2D<float> > U1,
		boost::shared_ptr<gpu_ptr_2D<float> > U2,
		boost::shared_ptr<gpu_ptr_2D<float> > U3) {
	GLuint U1tex = rd->getWaterTex();
	GLuint U2tex = rd->getWaterHUTex();
	GLuint U3tex = rd->getWaterHVTex();

	switch(water_render_mode) {
	case SWVisualizationContext::WATER_FRESNEL:
	case SWVisualizationContext::WATER_DEPTH:
		memcpyCudaToOgl(U1tex, 0, 0,
				address2D(U1->getRawPtr().ptr, U1->getRawPtr().pitch, 2, 2),
				U1->getRawPtr().pitch,
				(width)*sizeof(float), height);
		break;
	case SWVisualizationContext::WATER_VELOCITY:
		memcpyCudaToOgl(U1tex, 0, 0,
				address2D(U1->getRawPtr().ptr, U1->getRawPtr().pitch, 2, 2),
				U1->getRawPtr().pitch,
				(width)*sizeof(float), height);
		memcpyCudaToOgl(U2tex, 0, 0,
				address2D(U2->getRawPtr().ptr, U2->getRawPtr().pitch, 2, 2),
				U2->getRawPtr().pitch,
				(width)*sizeof(float), height);
		memcpyCudaToOgl(U3tex, 0, 0,
				address2D(U3->getRawPtr().ptr, U3->getRawPtr().pitch, 2, 2),
				U3->getRawPtr().pitch,
				(width)*sizeof(float), height);
		break;
	}
}

void SWVisualizationContext::setTexture(string filename) {
	try {
		boost::shared_ptr<util::PPMImage<float> > texture_texture;
		texture_texture = util::PPMImage<float>::read(filename.c_str());
		rd->setTexture(texture_texture);
	}
	catch (util::PNMImageExcpetion e) {
		std::cout << "Error reading texture:" << std::endl;
        std::cout << e.what() << std::endl;
        exit(-1);
    }
}

void SWVisualizationContext::setNormalMap(string filename) {
	try {
		boost::shared_ptr<util::PPMImage<float> > normal_texture;
		normal_texture = util::PPMImage<float>::read(filename.c_str());
		rd->setNormals(normal_texture);
	}
	catch (util::PNMImageExcpetion e) {
		std::cout << "Error reading normal map:" << std::endl;
		std::cout << e.what() << std::endl;
		exit(-1);
	}
}

void SWVisualizationContext::setBumpMap(string filename) {
	try {
		boost::shared_ptr<util::PPMImage<float> > bump_texture;
		bump_texture = util::PPMImage<float>::read(filename.c_str());
		rd->setBump(bump_texture);
	}
	catch (util::PNMImageExcpetion e) {
		std::cout << "Error reading bump map:" << std::endl;
		std::cout << e.what() << std::endl;
		exit(-1);
	}
}


