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

#ifndef CAMERATRANSFORM_H_
#define CAMERATRANSFORM_H_


struct CameraState { //Keeps track of our visualization state
	float x, y, z; //< position
	float u, v; //< horizontal and vertical angle of view direction

	CameraState(float x, float y, float z, float u, float v) {
		this->x = x;
		this->y = y;
		this->z = z;
		this->u = u;
		this->v = v;
	}
	CameraState() {};
};






class CameraTransform {
public:
	CameraTransform(CameraState& camera);

	const CameraState& getCameraState() { return camera; }
	const void setCameraState(CameraState c) { camera = c; }

	void rotate(float u=0.0f, float v=0.0f);
	void translate(float x=0.0f, float y=0.0f, float z=0.0f);

	void rotate(bool left, bool right, bool up, bool down, float h=0.0f);
	void translate(bool forward, bool backward, bool left, bool right, bool up, bool down, float h=0.0f);

private:
	CameraState camera;
};

#endif /* CAMERATRANSFORM_H_ */
