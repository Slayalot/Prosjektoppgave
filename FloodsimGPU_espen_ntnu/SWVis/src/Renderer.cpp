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

#define PREDEF_SKYBOX
//#define GLSL_FROM_FILE

#include "Renderer.h"

#include <iostream>
#include <limits>
#include <cmath>

#include "siut/gl_utils/GLSLtools.hpp"
#include "siut/io_utils/snarf.hpp"
#include "GLFuncs.h"

#ifdef PREDEF_SKYBOX
#include "skybox/skybox.h"
#endif
#ifndef GLSL_FROM_FILE
#include "glsl/Shaders.h"
#endif

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

using std::cout;
using std::min;
using std::max;

using namespace siut::gl_utils;
using namespace siut::io_utils;

using boost::shared_ptr;
using util::PPMImage;




namespace {
template<typename T>
inline void convert(typename boost::shared_ptr<util::PPMImage<T> > in, float* out, unsigned int pixels) {
#pragma omp parallel for schedule(static, 1)
	for (int i=0; i<(int) pixels; ++i) {
		out[3*i  ] = in->getRedData()[i];
		out[3*i+1] = in->getGreenData()[i];
		out[3*i+2] = in->getBlueData()[i];
	}
}

template<typename T>
inline void convert(float* in, typename boost::shared_ptr<util::PPMImage<T> > out, unsigned int pixels) {
#pragma omp parallel for schedule(static, 1)
	for (int i=0; i<(int) pixels; ++i) {
		out->getRedData()[i] = in[3*i  ];
		out->getGreenData()[i] = in[3*i+1];
		out->getBlueData()[i] = in[3*i+2];
	}
}

template <typename T>
inline void convert(typename boost::shared_ptr<util::PGMImage<T> > in, float* out, unsigned int pixels) {
#pragma omp parallel for schedule(static, 1)
	for (int i=0; i<(int) pixels; ++i) {
		out[i] = (float) in->getGrayData()[i];
	}
}
} //end namespace


Renderer::Renderer(unsigned int width, unsigned int height, float dx, float dy, float dz) {
	this->width = width;
	this->height = height;
	this->dx = dx;
	this->dy = dy;
	this->dz = dz;

	light[0] = -1.0f;
	light[1] = 1.0f;
	light[2] = -2.0f;

	genMesh(water_mesh, width, height, dx, dy);
	genMesh(landskape_mesh, width, height, dx, dy);

	initWaterFresnelShaders();
	initWaterDepthShaders();
	initWaterVelocityShaders();
	initLandskapeShaders();

	initTextures();
}

void Renderer::setTexture(boost::shared_ptr<util::PPMImage<float> >& texture) {
	float* tmp = new float[texture->getWidth()*texture->getHeight()*3];

	convert(texture, tmp, texture->getWidth()*texture->getHeight());
	landscape_tex = GLFuncs::newColorTexture(texture->getWidth(), texture->getHeight()); //FIXME: Very ugly, Does not free data...
    glBindTexture(GL_TEXTURE_2D, landscape_tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, texture->getWidth(), texture->getHeight(), GL_RGB, GL_FLOAT, tmp);
    glBindTexture(GL_TEXTURE_2D, 0);
    CHECK_GL_ERRORS();

	delete [] tmp;
}

void Renderer::setTextureFromHeightMap(float* height, int nx, int ny) {
	float* tmp = new float[nx*ny*3];
	float minimum = std::numeric_limits<float>::max();
	float maximum = std::numeric_limits<float>::min();

	for (int i=0; i<nx*ny; ++i) {
		minimum = min(minimum, height[i]);
		maximum = max(maximum, height[i]);
	}

	for (int i=0; i<nx*ny; ++i) {
		float color = (height[i]-minimum) / (maximum-minimum);
		tmp[3*i  ] = color;
		tmp[3*i+1] = color;
		tmp[3*i+2] = color;
	}

	landscape_tex = GLFuncs::newColorTexture(nx, ny); //FIXME: Very ugly, Does not free data...
    glBindTexture(GL_TEXTURE_2D, landscape_tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, nx, ny, GL_RGB, GL_FLOAT, tmp);
    glBindTexture(GL_TEXTURE_2D, 0);
    CHECK_GL_ERRORS();

	delete [] tmp;
}

void Renderer::setNormals(boost::shared_ptr<util::PPMImage<float> >& normal) {
	float* tmp = new float[normal->getWidth()*normal->getHeight()*3];

	convert(normal, tmp, normal->getWidth()*normal->getHeight());
	for (unsigned int i=0; i<normal->getWidth()*normal->getHeight()*3; ++i) tmp[i] -= 0.5f;
	normal_map = GLFuncs::newColorTexture(normal->getWidth(), normal->getHeight()); //FIXME: Very ugly, Does not free data...
    glBindTexture(GL_TEXTURE_2D, normal_map);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, normal->getWidth(), normal->getHeight(), GL_RGB, GL_FLOAT, tmp);
    glBindTexture(GL_TEXTURE_2D, 0);
    CHECK_GL_ERRORS();

	delete [] tmp;
}

void Renderer::setNormalsFromHeightMap(float* height, int nx, int ny) {
	int nnx = std::max(nx-2, 1);
	int nny = std::max(ny-2, 1);
	float* tmp = new float[nnx*nny*3];

	for (int j=0; j<nny; ++j) {
		for (int i=0; i<nnx; ++i) {
			int ii = std::min(i+1, nx-1);
			int iii = std::min(i+2, nx-1);
			int jj = std::min(j+1, ny-1);
			int jjj = std::min(j+2, ny-1);

			float n = height[jjj*nx+ ii];
			float s = height[  j*nx+ ii];
			float e = height[ jj*nx+iii];
			float w = height[ jj*nx+  i];

			//calculate the x, y and z components of the
			//normal using the cross product.
			//a x b = i(a2b3-a3b2) + j(a3b1-a1b3) +k(a1b2 -a2b1)
			//a = [2*dx, 0, dz*(e-w)]
			//b = [0, 2*dy, dz*(n-s)]
			float x = 2*dy*dz*(w-e);
			float y = 2*dx*dz*(s-n);
			float z = 4*dx*dy;

			//Normalize
			float l = sqrt(x*x+y*y+z*z);
			x /= l;
			y /= l;
			z /= l;

			tmp[3*(j*nnx+i)  ] = x;
			tmp[3*(j*nnx+i)+1] = y;
			tmp[3*(j*nnx+i)+2] = z;
		}
	}

	normal_map = GLFuncs::newColorTexture(nnx, nny); //FIXME: Very ugly, Does not free data...
    glBindTexture(GL_TEXTURE_2D, normal_map);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, nnx, nny, GL_RGB, GL_FLOAT, tmp);
    glBindTexture(GL_TEXTURE_2D, 0);
    CHECK_GL_ERRORS();

	delete [] tmp;
}
void Renderer::setBump(boost::shared_ptr<util::PPMImage<float> >& bump) {
	float* tmp = new float[bump->getWidth()*bump->getHeight()*3];

	//Bump map
	convert(bump, tmp, bump->getWidth()*bump->getHeight());
	for (unsigned int i=0; i<bump->getWidth()*bump->getHeight()*3; ++i) tmp[i] -= 0.5f;
	bump_map = GLFuncs::newColorTexture(bump->getWidth(), bump->getHeight()); //FIXME: Very ugly, Does not free data...
    glBindTexture(GL_TEXTURE_2D, bump_map); //reset texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, bump->getWidth(), bump->getHeight(), GL_RGB, GL_FLOAT, tmp);
    glBindTexture(GL_TEXTURE_2D, 0);
    CHECK_GL_ERRORS();

	delete [] tmp;
}

void Renderer::setSkybox(shared_ptr<PPMImage<float> >& west,
			shared_ptr<PPMImage<float> >& east,
			shared_ptr<PPMImage<float> >& north,
			shared_ptr<PPMImage<float> >& south,
			shared_ptr<PPMImage<float> >& down,
			shared_ptr<PPMImage<float> >& up) {
	skybox = GLFuncs::newCubeMap(west->getWidth(), west->getHeight());
    CHECK_GL_ERRORS();
    glBindTexture(GL_TEXTURE_CUBE_MAP, skybox);

	unsigned int width = west->getWidth();
	unsigned int height = west->getHeight();

	float* tmp = new float[width*height*3];

	convert(north, tmp, width*height);
    glTexSubImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, 0, 0, width, height, GL_RGB, GL_FLOAT, tmp);
	convert(south, tmp, width*height);
    glTexSubImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, 0, 0, width, height, GL_RGB, GL_FLOAT, tmp);
	convert(east, tmp, width*height);
    glTexSubImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, 0, 0, width, height, GL_RGB, GL_FLOAT, tmp);
	convert(west, tmp, width*height);
    glTexSubImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, 0, 0, width, height, GL_RGB, GL_FLOAT, tmp);
	convert(down, tmp, width*height);
    glTexSubImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, 0, 0, width, height, GL_RGB, GL_FLOAT, tmp);
	convert(up, tmp, width*height);
    glTexSubImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, 0, 0, width, height, GL_RGB, GL_FLOAT, tmp);

	delete [] tmp;

    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
    CHECK_GL_ERRORS();
}

void Renderer::initTextures() {
	water_height_map = GLFuncs::newGrayTexture(width, height);
	water_hu_map = GLFuncs::newGrayTexture(width, height);
	water_hv_map = GLFuncs::newGrayTexture(width, height);
    landscape_height_map = GLFuncs::newGrayTexture(width, height);

    /**
     * Skybox
     */
#ifdef PREDEF_SKYBOX
    skybox = GLFuncs::newCubeMap(skybox_width, skybox_height);
    CHECK_GL_ERRORS();
    glBindTexture(GL_TEXTURE_CUBE_MAP, skybox);
    glTexSubImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, 0, 0, skybox_width, skybox_height, GL_RGB, GL_UNSIGNED_BYTE, skybox_north);
    glTexSubImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, 0, 0, skybox_width, skybox_height, GL_RGB, GL_UNSIGNED_BYTE, skybox_south);
    glTexSubImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, 0, 0, skybox_width, skybox_height, GL_RGB, GL_UNSIGNED_BYTE, skybox_east);
    glTexSubImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, 0, 0, skybox_width, skybox_height, GL_RGB, GL_UNSIGNED_BYTE, skybox_west);
    glTexSubImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, 0, 0, skybox_width, skybox_height, GL_RGB, GL_UNSIGNED_BYTE, skybox_down);
    glTexSubImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, 0, 0, skybox_width, skybox_height, GL_RGB, GL_UNSIGNED_BYTE, skybox_up);
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
    CHECK_GL_ERRORS();
#endif

    CHECK_GL_ERRORS();
}

void Renderer::initWaterFresnelShaders() {
	GLint loc;
	GLuint vshader, fshader;
	try {
#ifdef GLSL_FROM_FILE
		vshader = compileShader(snarfFile("src/glsl/Water_fresnel_vertex.glsl"), GL_VERTEX_SHADER);
		fshader = compileShader(snarfFile("src/glsl/Water_fresnel_fragment.glsl"), GL_FRAGMENT_SHADER);
#else
		vshader = compileShader(water_fresnel_vertex::source, GL_VERTEX_SHADER);
		fshader = compileShader(water_fresnel_fragment::source, GL_FRAGMENT_SHADER);
#endif
	}
	catch(std::runtime_error e) {
		cout << e.what();
	}


	water_fresnel_program = glCreateProgram();
	glAttachShader(water_fresnel_program, vshader);
	glAttachShader(water_fresnel_program, fshader);
	linkProgram(water_fresnel_program);
	glUseProgram(water_fresnel_program);

	loc = glGetUniformLocation(water_fresnel_program, "water_height_map");
	glUniform1i(loc, 0);
	loc = glGetUniformLocation(water_fresnel_program, "skybox");
	glUniform1i(loc, 1);
	loc = glGetUniformLocation(water_fresnel_program, "landscape_tex");
	glUniform1i(loc, 2);
	loc = glGetUniformLocation(water_fresnel_program, "height_map");
	glUniform1i(loc, 3);
	loc = glGetUniformLocation(water_fresnel_program, "normal_map");
	glUniform1i(loc, 4);
	loc = glGetUniformLocation(water_fresnel_program, "du");
	glUniform1f(loc, 1.0f/(float) width);
	loc = glGetUniformLocation(water_fresnel_program, "dv");
	glUniform1f(loc, 1.0f/(float) height);
	loc = glGetUniformLocation(water_fresnel_program, "dx");
	glUniform1f(loc, dx);
	loc = glGetUniformLocation(water_fresnel_program, "dy");
	glUniform1f(loc, dy);
	loc = glGetUniformLocation(water_fresnel_program, "dz");
	glUniform1f(loc, dz);
	loc = glGetUniformLocation(water_fresnel_program, "L"); //< Light vector
	glUniform3f(loc, light[0], light[1], light[2]);

	glUseProgram(0);
    CHECK_GL_ERRORS();
}


void Renderer::initWaterDepthShaders() {
	GLint loc;
	GLuint vshader, fshader;
	try {
#ifdef GLSL_FROM_FILE
		vshader = compileShader(snarfFile("src/glsl/Water_depth_vertex.glsl"), GL_VERTEX_SHADER);
		fshader = compileShader(snarfFile("src/glsl/Water_depth_fragment.glsl"), GL_FRAGMENT_SHADER);
#else
		vshader = compileShader(water_depth_vertex::source, GL_VERTEX_SHADER);
		fshader = compileShader(water_depth_fragment::source, GL_FRAGMENT_SHADER);
#endif
	}
	catch(std::runtime_error e) {
		cout << e.what();
	}

	water_depth_program = glCreateProgram();
	glAttachShader(water_depth_program, vshader);
	glAttachShader(water_depth_program, fshader);
	linkProgram(water_depth_program);
	glUseProgram(water_depth_program);

	loc = glGetUniformLocation(water_depth_program, "water_height_map");
	glUniform1i(loc, 0);
	loc = glGetUniformLocation(water_depth_program, "height_map");
	glUniform1i(loc, 1);
	loc = glGetUniformLocation(water_depth_program, "dz");
	glUniform1f(loc, dz);
	loc = glGetUniformLocation(water_depth_program, "h_min");
	glUniform1f(loc, 0);
	loc = glGetUniformLocation(water_depth_program, "h_max");
	glUniform1f(loc, dx);

	glUseProgram(0);
    CHECK_GL_ERRORS();
}

void Renderer::initWaterVelocityShaders() {
	GLint loc;
	GLuint vshader, fshader;
	try {
#ifdef GLSL_FROM_FILE
		vshader = compileShader(snarfFile("src/glsl/Water_velocity_vertex.glsl"), GL_VERTEX_SHADER);
		fshader = compileShader(snarfFile("src/glsl/Water_velocity_fragment.glsl"), GL_FRAGMENT_SHADER);
#else
		vshader = compileShader(water_velocity_vertex::source, GL_VERTEX_SHADER);
		fshader = compileShader(water_velocity_fragment::source, GL_FRAGMENT_SHADER);
#endif
	}
	catch(std::runtime_error e) {
		cout << e.what();
	}

	water_velocity_program = glCreateProgram();
	glAttachShader(water_velocity_program, vshader);
	glAttachShader(water_velocity_program, fshader);
	linkProgram(water_velocity_program);
	glUseProgram(water_velocity_program);

	loc = glGetUniformLocation(water_velocity_program, "water_height_map");
	glUniform1i(loc, 0);
	loc = glGetUniformLocation(water_velocity_program, "water_hu_map");
	glUniform1i(loc, 1);
	loc = glGetUniformLocation(water_velocity_program, "water_hv_map");
	glUniform1i(loc, 2);
	loc = glGetUniformLocation(water_velocity_program, "height_map");
	glUniform1i(loc, 3);
	loc = glGetUniformLocation(water_velocity_program, "dz");
	glUniform1f(loc, dz);
	loc = glGetUniformLocation(water_velocity_program, "h_min");
	glUniform1f(loc, 0);
	loc = glGetUniformLocation(water_velocity_program, "h_max");
	glUniform1f(loc, 0.5f*dx);

	glUseProgram(0);
    CHECK_GL_ERRORS();
}

void Renderer::initLandskapeShaders() {
	GLint loc;
	GLuint vshader, fshader;
	try {
#ifdef GLSL_FROM_FILE
		vshader = compileShader(snarfFile("src/glsl/Landscape.vglsl"), GL_VERTEX_SHADER);
		fshader = compileShader(snarfFile("src/glsl/Landscape.fglsl"), GL_FRAGMENT_SHADER);
#else
		vshader = compileShader(landscape_vertex::source, GL_VERTEX_SHADER);
		fshader = compileShader(landscape_fragment::source, GL_FRAGMENT_SHADER);
#endif

	}
	catch(std::runtime_error e) {
		cout << e.what();
	}

	landscape_program = glCreateProgram();
	glAttachShader(landscape_program, vshader);
	glAttachShader(landscape_program, fshader);
	linkProgram(landscape_program);
	glUseProgram(landscape_program);

	loc = glGetUniformLocation(landscape_program, "skybox");
	glUniform1i(loc, 4);
	loc = glGetUniformLocation(landscape_program, "bump_map");
	glUniform1i(loc, 3);
	loc = glGetUniformLocation(landscape_program, "normal_map");
	glUniform1i(loc, 2);
	loc = glGetUniformLocation(landscape_program, "landscape_tex");
	glUniform1i(loc, 1);
	loc = glGetUniformLocation(landscape_program, "height_map");
	glUniform1i(loc, 0);
	loc = glGetUniformLocation(landscape_program, "du");
	glUniform1f(loc, 1.0f/(float) width);
	loc = glGetUniformLocation(landscape_program, "dv");
	glUniform1f(loc, 1.0f/(float) height);
	loc = glGetUniformLocation(landscape_program, "dx");
	glUniform1f(loc, dx);
	loc = glGetUniformLocation(landscape_program, "dy");
	glUniform1f(loc, dy);
	loc = glGetUniformLocation(landscape_program, "dz");
	glUniform1f(loc, dz);
	loc = glGetUniformLocation(landscape_program, "L"); //< Light vector
	glUniform3f(loc, light[0], light[1], light[2]);

	glUseProgram(0);
    CHECK_GL_ERRORS();
}


/**
  * Render skybox
  */
void Renderer::renderSkybox() {
	float modmat[16];

	//Find rotation matrix
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glGetFloatv(GL_MODELVIEW_MATRIX, &modmat[0]);
	modmat[12] = 0;
	modmat[13] = 0;
	modmat[14] = 0;
	glLoadMatrixf(&modmat[0]);
	CHECK_GL_ERRORS();

	//Vertices
	float v[8][3] = {{-0.5f, -0.5f, 0.5f},
					{0.5f, -0.5f, 0.5f},
					{0.5f, 0.5f, 0.5f},
					{-0.5f, 0.5f, 0.5f},
					{-0.5f, -0.5f, -0.5f},
					{0.5f, -0.5f, -0.5f},
					{0.5f, 0.5f, -0.5f},
					{-0.5f, 0.5f, -0.5f}};
	unsigned int i[24] = {
		0, 1, 2, 3,
		1, 5, 6, 2,
		5, 4, 7, 6,
		4, 0, 3, 7,
		0, 4, 5, 1,
		2, 6, 7, 3};
	float scale = 0.5f*max(dx*width, dy*height);

	//Render cube without depth test
    glDisable(GL_DEPTH_TEST);
    glActiveTexture(GL_TEXTURE0);
    glEnable(GL_TEXTURE_CUBE_MAP);
    glBindTexture(GL_TEXTURE_CUBE_MAP, skybox);

	glBegin(GL_QUADS);
	for (int k=0; k<24; ++k) {
		glTexCoord3f(v[i[k]][0], v[i[k]][1], v[i[k]][2]);
		glVertex3f(scale*v[i[k]][0], scale*v[i[k]][1], scale*v[i[k]][2]);
	}
	glEnd();

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
    glDisable(GL_TEXTURE_CUBE_MAP);
    glEnable(GL_DEPTH_TEST);
	glClear(GL_DEPTH_BUFFER_BIT);

	glPopMatrix();
	CHECK_GL_ERRORS();
}

void Renderer::renderLandscape() {
	/**
	  * Render landscape
	  */
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, landskape_mesh.indices); //Indices
	glBindBuffer(GL_ARRAY_BUFFER, landskape_mesh.vertices); //Vertices
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glVertexPointer(3, GL_FLOAT, 5*sizeof(float), 0);
	glTexCoordPointer(2, GL_FLOAT, 5*sizeof(float), (const GLubyte *)NULL + 3*sizeof(float));

    glUseProgram(landscape_program);
	CHECK_GL_ERRORS();
    glActiveTexture(GL_TEXTURE0);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, landscape_height_map);

    glActiveTexture(GL_TEXTURE1);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, landscape_tex);

    glActiveTexture(GL_TEXTURE2);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, normal_map);

    glActiveTexture(GL_TEXTURE3);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, bump_map);

	glActiveTexture(GL_TEXTURE4);
	glEnable(GL_TEXTURE_CUBE_MAP);
	glBindTexture(GL_TEXTURE_CUBE_MAP, skybox);

	glDrawElements(GL_QUAD_STRIP, landskape_mesh.n_vertices, GL_UNSIGNED_INT, 0);

    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
    glDisable(GL_TEXTURE_CUBE_MAP);

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);

	glUseProgram(0);
	CHECK_GL_ERRORS();
}


void Renderer::renderWaterFresnel() {
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, water_mesh.indices); //Indices
	glBindBuffer(GL_ARRAY_BUFFER, water_mesh.vertices); //Vertices
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glVertexPointer(3, GL_FLOAT, 5*sizeof(float), 0);
	glTexCoordPointer(2, GL_FLOAT, 5*sizeof(float), (const GLubyte *)NULL + 3*sizeof(float));

	glUseProgram(water_fresnel_program);
	glActiveTexture(GL_TEXTURE0);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, water_height_map);

	glActiveTexture(GL_TEXTURE1);
	glEnable(GL_TEXTURE_CUBE_MAP);
	glBindTexture(GL_TEXTURE_CUBE_MAP, skybox);

	glActiveTexture(GL_TEXTURE2);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, landscape_tex);

	glActiveTexture(GL_TEXTURE3);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, landscape_height_map);

	glActiveTexture(GL_TEXTURE4);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, normal_map);

	glDrawElements(GL_QUAD_STRIP, water_mesh.n_vertices, GL_UNSIGNED_INT, 0);

	glActiveTexture(GL_TEXTURE4);
	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);

	glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
	glDisable(GL_TEXTURE_CUBE_MAP);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);
	glUseProgram(0);
	CHECK_GL_ERRORS();

	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void Renderer::renderWaterDepth() {
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, water_mesh.indices); //Indices
	glBindBuffer(GL_ARRAY_BUFFER, water_mesh.vertices); //Vertices
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glVertexPointer(3, GL_FLOAT, 5*sizeof(float), 0);
	glTexCoordPointer(2, GL_FLOAT, 5*sizeof(float), (const GLubyte *)NULL + 3*sizeof(float));

	glUseProgram(water_depth_program);
	glActiveTexture(GL_TEXTURE0);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, water_height_map);

	glActiveTexture(GL_TEXTURE1);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, landscape_height_map);

	glDrawElements(GL_QUAD_STRIP, water_mesh.n_vertices, GL_UNSIGNED_INT, 0);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);
	glUseProgram(0);
	CHECK_GL_ERRORS();

	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void Renderer::renderWaterVelocity() {
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, water_mesh.indices); //Indices
	glBindBuffer(GL_ARRAY_BUFFER, water_mesh.vertices); //Vertices
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glVertexPointer(3, GL_FLOAT, 5*sizeof(float), 0);
	glTexCoordPointer(2, GL_FLOAT, 5*sizeof(float), (const GLubyte *)NULL + 3*sizeof(float));

	glUseProgram(water_velocity_program);
	glActiveTexture(GL_TEXTURE0);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, water_height_map);

	glActiveTexture(GL_TEXTURE1);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, water_hu_map);

	glActiveTexture(GL_TEXTURE2);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, water_hv_map);

	glActiveTexture(GL_TEXTURE3);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, landscape_height_map);

	glDrawElements(GL_QUAD_STRIP, water_mesh.n_vertices, GL_UNSIGNED_INT, 0);

	glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);
	glUseProgram(0);
	CHECK_GL_ERRORS();

	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void Renderer::renderString(std::string& strs) {
	unsigned int window_width = glutGet(GLUT_WINDOW_WIDTH);
	unsigned int window_height = glutGet(GLUT_WINDOW_HEIGHT);
	unsigned int text_width = 0;
	unsigned int tmp_text_width = 0;
	unsigned int text_height = 0;
	unsigned int lines = 0;

	for (unsigned int i=0; i<strs.length(); ++i) {
		if (strs.at(i) == '\n') {
			text_width = std::max(text_width, tmp_text_width);
			tmp_text_width = 0;
			lines++;
		}
		else {
			tmp_text_width += glutBitmapWidth(GLUT_BITMAP_8_BY_13, strs[i]);
		}
	}
	text_width = std::max(text_width, tmp_text_width);
	lines++;
	text_height = (lines)*13;

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0, window_width, 0, window_height, -1.0, 1.0);

	glDisable(GL_DEPTH_TEST);

	glColor4f(0.0f, 0.0f, 0.0f, 0.6f);
	glBegin(GL_QUADS);
	glVertex2f(0.0f, 0.0f);
	glVertex2f(10.0f+text_width, 0.0f);
	glVertex2f(10.0f+text_width, 5.0f+text_height);
	glVertex2f(0.0f, 5.0f+text_height);
	glEnd();

	glColor4f(0.0f, 1.0f, 0.0f, 1.0f);
	--lines;
	glRasterPos2f(5.0f, 5.0f+lines*13.0f);
	for (unsigned int i=0; i<strs.length(); ++i) {
		if (strs.at(i) == '\n') {
			--lines;
			glRasterPos2f(5.0f, 5.0f+lines*13.0f);
		}
		else {
			glutBitmapCharacter(GLUT_BITMAP_8_BY_13 , strs[i]);
		}
	}

	glEnable(GL_DEPTH_TEST);
	glColor3f(1.0f, 1.0f, 1.0f);

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();

	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
}

void Renderer::genMesh(mesh& mesh, unsigned int width, unsigned int height, float dx, float dy) {
	int w = width+1;
	int h = height+1;

	{ //First, vertices
		std::vector<float> vca;
		unsigned int size;

		//Generate vertex and texture coordinates
		for(int y=0; y<h; y++) {
			for(int x=0; x<w; x++) {
				//Vertex coordinates
				vca.push_back((float) x*dx); //<X
				vca.push_back(0.0f);         //<Y
				vca.push_back((float) y*dy); //<Z

				//Texture coordinates
				vca.push_back((x+0.5f)/(float) (w)); //<S
				vca.push_back((y+0.5f)/(float) (h)); //<T
			}
		}

		size = vca.size()*sizeof(float);

		//Generate vbo, and upload newly created coordinates.
		glGenBuffers(1, &mesh.vertices);
		glBindBuffer(GL_ARRAY_BUFFER, mesh.vertices);
		glBufferData(GL_ARRAY_BUFFER, size, &vca[0], GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	{ //Then indices
		std::vector<unsigned int> ica;

		for(int y=0; y<h-1; y++) {
			for(int x=0; x<w; x++) { //Forward strip
				ica.push_back(x+(y+1)*(w));
				ica.push_back(x+ y   *(w));
			}
			ica.push_back((y+2)*(w)-1); //Reverse (degenerate)
			ica.push_back((y+2)*(w)-1); //Reverse (degenerate)
			y++;
			if (y<h-1) {
				for(int x=w-1; x>=0; --x) { //Backward strip
					ica.push_back(x+ y   *(w));
					ica.push_back(x+(y+1)*(w));
				}
				ica.push_back((y+1)*(w)); //Reverse (degenerate)
				ica.push_back((y+1)*(w)); //Reverse (degenerate)
			}
		}

		glGenBuffers(1, &mesh.indices);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.indices);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, ica.size()*sizeof(unsigned int), &ica[0], GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		mesh.n_vertices = ica.size();
	}
}

