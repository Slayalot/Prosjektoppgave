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

#ifndef SWVISUALIZATIONCONTEXT_HPP_
#define SWVISUALIZATIONCONTEXT_HPP_

#include <GL/glew.h>
#include <GL/gl.h>

#include <string>
#include <sstream>
#include <vector>


#include <boost/shared_ptr.hpp>

#include "KPSimulator.h"
#include "CameraTransform.h"
#include "ScreenshotCreator.h"
#include "Renderer.h"


class SimulatorWrapper {
public:
	virtual void step() = 0;
	virtual float getTime() = 0;
	virtual float getDt() = 0;
	virtual int getTimeStep() = 0;
	virtual boost::shared_ptr<gpu_ptr_2D<float> > getU1() = 0;
	virtual boost::shared_ptr<gpu_ptr_2D<float> > getU2() = 0;
	virtual boost::shared_ptr<gpu_ptr_2D<float> > getU3() = 0;
	virtual boost::shared_ptr<gpu_ptr_2D<float> > getB() = 0;
};

class SWVisualizationContext {
public:
	enum WATER_RENDER_TYPE {
		WATER_FRESNEL,
		WATER_DEPTH,
		WATER_VELOCITY,
	};

public:
	SWVisualizationContext();
	SWVisualizationContext(KPInitialConditions& init, float vertical_scale=1.0);

	boost::shared_ptr<CameraTransform> getCameraTransform();
	boost::shared_ptr<Renderer> getRenderer();
	const WATER_RENDER_TYPE& getWaterRenderType();

	static void initGlut(int argc, char** argv);

	bool renderFunc(SimulatorWrapper& sim);
	void renderScene(float elapsed);
	void keyboardFunc(unsigned char key, int x, int y, int modifiers=0);
	void keyboardUpFunc(unsigned char key, int x, int y, int modifiers=0);
	void specialFunc(unsigned char key);
	void mouseFunc(int b, int s, int x, int y);
	void passiveMotionFunc(int x, int y);
	void reshapeFunc(int width, int height);

	void updateBData(boost::shared_ptr<gpu_ptr_2D<float> > B);
	void updateUData(boost::shared_ptr<gpu_ptr_2D<float> > U1,
			boost::shared_ptr<gpu_ptr_2D<float> > U2,
			boost::shared_ptr<gpu_ptr_2D<float> > U3);

	void setTexture(std::string filename);
	void setNormalMap(std::string filename);
	void setBumpMap(std::string filename);

	bool getSimulate() { return simulate; }

	static void memcpyCudaToOgl(GLuint dst, size_t w_offset, size_t h_offset,
		float* src, size_t pitch, size_t width_in_bytes, size_t height);

private:
	void init(size_t width, size_t height, float dx, float dy, float dz);
	void initGLState();
	void setProjectionMatrix();
	void updateData(SimulatorWrapper& sim);
	void runSimulation(SimulatorWrapper& sim);

	void playRecord();
	void addRecord(float sim_time);
	void saveRecorder();
	void readRecorder();
	


private:
	boost::shared_ptr<CameraTransform> ct;
	boost::shared_ptr<Renderer> rd;
	boost::shared_ptr<ScreenshotCreator> sc;

	size_t screen_width;
	size_t screen_height;
	size_t screen_center_x;
	size_t screen_center_y;
	size_t width;
	size_t height;
	float dx, dy, dz;
	float camera_speed;
	double next_passivemotion_time;
	double next_render_time;
	double last_render_time;
	double next_simulation_time;
	double fps_rate;
	float ips_avg, fps_avg, speedup_avg, dt_avg;
	long iters;
	std::string osd;

	bool keyDown[256];
	bool capture_mouse;
	bool record;
	bool playback;
	bool simulate;
	float sim_speed;
	float sim_time;
	bool screenshots;
	bool u_data_changed;
	bool b_data_changed;
	bool wireframe;
	bool skybox;
	bool rotate;
	float rotate_degree;
	bool step;

	//Camera stuff.
	float x_aspect;
	float y_aspect;
	float zoom;
	float fz0;
	float fz1;

	int playback_position;

	struct SWVisRecord {
		CameraState camera;
		WATER_RENDER_TYPE water_render_mode;
		bool wireframe;
		float rotate_degree;
		float sim_time;
		std::string osd;
	};

	std::vector<SWVisRecord> recorder;
	std::string recorder_filename;

	WATER_RENDER_TYPE water_render_mode;
};












#endif /* SWVISUALIZATIONCONTEXT_HPP_ */
