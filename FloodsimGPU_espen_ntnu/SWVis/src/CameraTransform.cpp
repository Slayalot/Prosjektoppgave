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

#include "CameraTransform.h"

#include <cmath>

#include "GLFuncs.h"

CameraTransform::CameraTransform(CameraState& camera) {
	this->camera = camera;
}

void CameraTransform::rotate(float u, float v) {
	camera.u += u;
	camera.v += v;

	//We don't want to flip over...
	if (camera.v >  90.0f) {camera.v =  90.0f;}
	if (camera.v < -90.0f) {camera.v = -90.0f;}

	glRotatef(camera.v, 1.0f, 0.0f, 0.0f);
	glRotatef(camera.u, 0.0f, 1.0f, 0.0f);
}

void CameraTransform::rotate(bool left, bool right, bool up, bool down, float h) {
	float u = 0.0f;
	float v = 0.0f;;

	if (left) u -= h;
	if (right) u += h;
	if (up) v -= h;
	if (down) v += h;
	rotate(u, v);
}

void CameraTransform::translate(float x, float y, float z) {
	camera.x += x;
	camera.y += y;
	camera.z += z;

	glTranslatef(camera.x, camera.y, camera.z);
}

void CameraTransform::translate(bool forward, bool backward, bool left, bool right, bool up, bool down, float h) {
	static const float degToRad = 3.14159265f/180.0f;
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;
	if (forward || backward) { //Forward/Backward
		float tmp_x = h;
		float tmp_y = h;
		float tmp_z = h;

		tmp_x *= -sinf(camera.u*degToRad);
		tmp_x *= cosf(camera.v*degToRad);
		tmp_y *= sinf(camera.v*degToRad);
		tmp_z *= cosf(camera.u*degToRad);
		tmp_z *= cosf(camera.v*degToRad);

		if (backward) {
			x -= tmp_x;
			y -= tmp_y;
			z -= tmp_z;
		}
		else {
			x += tmp_x;
			y += tmp_y;
			z += tmp_z;
		}
	}
	if (up || down) { //Up, down
		float tmp_x = 0.5f*h;
		float tmp_y = 0.5f*h;
		float tmp_z = 0.5f*h;

		tmp_x *= sinf(camera.u*degToRad);
		tmp_x *= -sinf(camera.v*degToRad);
		tmp_y *= -cosf(camera.v*degToRad);
		tmp_z *= cosf(camera.u*degToRad);
		tmp_z *= sinf(camera.v*degToRad);

		if (down) {
			x -= tmp_x;
			y -= tmp_y;
			z -= tmp_z;
		}
		else {
			x += tmp_x;
			y += tmp_y;
			z += tmp_z;
		}
	}
	if (left || right) { //left/right
		float tmp_x = 0.5f*h;
		float tmp_y = 0.5f*h;
		float tmp_z = 0.5f*h;

		tmp_x *= cosf(camera.u*degToRad);
		tmp_y *= 0;
		tmp_z *= sinf(camera.u*degToRad);

		if (right) {
			x -= tmp_x;
			y -= tmp_y;
			z -= tmp_z;
		}
		else {
			x += tmp_x;
			y += tmp_y;
			z += tmp_z;
		}
	}

	translate(x, y, z);
}

