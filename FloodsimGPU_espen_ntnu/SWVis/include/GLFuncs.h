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

#ifndef GLFUNCS_H__
#define GLFUNCS_H__

#include <GL/glew.h>
#include <GL/glut.h>
#ifdef FREEGLUT
#include <GL/freeglut_ext.h>
#else
#include <cstdlib>
#endif
#include <GL/gl.h>

namespace GLFuncs {
	void checkGLErrors(const char* file, unsigned int line);

	GLuint newGrayTexture(size_t width, size_t height);
	GLuint newColorTexture(size_t width, size_t height);
	void deleteTexture(GLuint& tex);

	GLuint newBufferObject(size_t width, size_t height);
	void deleteBufferObject(GLuint& bo);

	GLuint newCubeMap(size_t width, size_t height);
	void deleteCubeMap(GLuint& tex);
}

#define CHECK_GL_ERRORS() GLFuncs::checkGLErrors(__FILE__, __LINE__)

#endif
