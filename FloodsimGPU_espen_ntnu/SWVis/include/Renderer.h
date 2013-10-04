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

#ifndef RENDERER_H_
#define RENDERER_H_

#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include "GLFuncs.h"
#include "PNMImage.hpp"

class Renderer {
public:
	Renderer(unsigned int width, unsigned int height, float dx, float dy, float dz);

	inline const GLuint& getWaterTex() { return water_height_map; }
	inline const GLuint& getWaterHUTex() { return water_hu_map; }
	inline const GLuint& getWaterHVTex() { return water_hv_map; }
	inline const GLuint& getBottomTex() { return landscape_height_map; }
	inline const GLuint& getSkybox() { return skybox; }

	void setTexture(boost::shared_ptr<util::PPMImage<float> >& texture);
	void setTextureFromHeightMap(float* height, int nx, int ny);
	void setNormals(boost::shared_ptr<util::PPMImage<float> >& normals);
	void setNormalsFromHeightMap(float* height, int nx, int ny);
	void setBump(boost::shared_ptr<util::PPMImage<float> >& bump);
	void setSkybox(boost::shared_ptr<util::PPMImage<float> >& west,
			boost::shared_ptr<util::PPMImage<float> >& east,
			boost::shared_ptr<util::PPMImage<float> >& north,
			boost::shared_ptr<util::PPMImage<float> >& south,
			boost::shared_ptr<util::PPMImage<float> >& bottom,
			boost::shared_ptr<util::PPMImage<float> >& top);

	void renderSkybox();
	void renderLandscape();
	void renderWaterFresnel();
	void renderWaterDepth();
	void renderWaterVelocity();
	void renderString(std::string& strs);

private:
	struct mesh {
		GLuint vertices; //< Vertex buffer object with triangulation of landscape
		GLuint indices; //< Vertex buffer object with triangulation of landscape
		size_t n_vertices;
	};
	static void genMesh(mesh& mesh, unsigned int width, unsigned int height, float dx, float dy);

	void initWaterFresnelShaders();
	void initWaterDepthShaders();
	void initWaterVelocityShaders();
	void initLandskapeShaders();

	void initTextures();

private:

	size_t width;
	size_t height;
	mesh water_mesh, landskape_mesh;
	GLuint water_height_map, water_hu_map, water_hv_map, landscape_height_map, skybox, landscape_tex, normal_map, bump_map;
	GLuint water_fresnel_program, water_depth_program, water_velocity_program, landscape_program;
	float light[3];

	float dx, dy, dz;
};

#endif /* RENDERER_H_ */
