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


uniform sampler2D height_map;
uniform sampler2D normal_map;
uniform float du; //Texture coordinates x
uniform float dv; // --"-- y
uniform float dx; 
uniform float dy; 
uniform float dz;

varying vec3 N;
varying vec3 v;
varying vec3 p;

void main(void) {
	float c; 
	vec4 position;
	
	//Look up textures
	c = texture2D(height_map, vec2(gl_MultiTexCoord0.x, gl_MultiTexCoord0.y)).x;
	
	//Displace the vertex according to height map
	position = gl_Vertex;
	position.y = c*dz;
	
	//Set object space variables for lighting
	p = position.xyz; //< vertex position
	N = texture2D(normal_map, gl_MultiTexCoord0.xy).xzy; //< normal
	vec4 viewer_h = gl_ModelViewMatrixInverse*vec4(0.0, 0.0, 0.0, 1.0);
	v = (1.0/viewer_h.w)*viewer_h.xyz; //< Viewer position
	
	//Set output vertex position
	gl_Position = gl_ModelViewProjectionMatrix*position;
	
	//Set texture coordinates
	gl_TexCoord[0] = gl_MultiTexCoord0;
}
