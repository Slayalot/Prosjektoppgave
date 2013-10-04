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

#ifndef COMMON_VISUALIZATION_HPP_
#define COMMON_VISUALIZATION_HPP_

#include <iostream>
#include <string>
#include <iomanip>
#include <vector>

#include "PNMImage.hpp"
#include "SWVisualizationContext.h"

#include <GL/glew.h>
#include <GL/glut.h>
#ifdef FREEGLUT
#include <GL/freeglut_ext.h>
#else
#include <cstdlib>
#endif
#include <GL/gl.h>

boost::shared_ptr<SWVisualizationContext> setup_vis(KPInitialConditions& init,
										std::string texture_filename,
										std::string normal_filename,
										std::string bump_filename, float vertical_scale) {
	boost::shared_ptr<SWVisualizationContext> vis;

	//Set up visualizer
    vis.reset(new SWVisualizationContext(init, vertical_scale));

	if (texture_filename.compare("") != 0) vis->setTexture(texture_filename);
	else vis->getRenderer()->setTextureFromHeightMap(init.getB(), init.getBNx(), init.getBNy());
	if (normal_filename.compare("") != 0) vis->setNormalMap(normal_filename);
	else vis->getRenderer()->setNormalsFromHeightMap(init.getB(), init.getBNx(), init.getBNy());
	if (bump_filename.compare("") != 0) vis->setBumpMap(bump_filename);

	return vis;
}




#endif /* COMMON_VISUALIZATION_HPP_ */
