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


uniform sampler2D water_height_map;
uniform sampler2D height_map;
uniform sampler2D normal_map;
uniform float du; //Texture coordinates x
uniform float dv; // --"-- y
uniform float dx; 
uniform float dy; 
uniform float dz;

varying vec3 N_water, N_landscape;
varying vec3 v;
varying vec3 p;
varying float h;

void main(void) {
	float n, s, e, w, c; //< north, south, east, west, center
	vec4 position;
	
	//Look up textures
	n = texture2D(water_height_map, vec2(gl_MultiTexCoord0.x   , gl_MultiTexCoord0.y+dv)).x;
	s = texture2D(water_height_map, vec2(gl_MultiTexCoord0.x   , gl_MultiTexCoord0.y-dv)).x;
	e = texture2D(water_height_map, vec2(gl_MultiTexCoord0.x+du, gl_MultiTexCoord0.y   )).x;
	w = texture2D(water_height_map, vec2(gl_MultiTexCoord0.x-du, gl_MultiTexCoord0.y   )).x;
	c = texture2D(water_height_map, vec2(gl_MultiTexCoord0.x   , gl_MultiTexCoord0.y   )).x;
	h = dz*(c-texture2D(height_map, vec2(gl_MultiTexCoord0.x, gl_MultiTexCoord0.y)).x); //water depth
	
	//Displace the vertex according to height map
	position = gl_Vertex;
	position.y = c*dz;
	
	//Set object space variables for lighting
	p = position.xyz; //< vertex position
	vec3 a = vec3(2*dx, dz*(e-w), 0.0);
	vec3 b = vec3(0.0, dz*(n-s), 2*dy);
	N_water = cross(a, b); //< normal
	N_landscape = texture2D(normal_map, gl_MultiTexCoord0.xy).xzy;
	vec4 viewer_h = gl_ModelViewMatrixInverse*vec4(0.0, 0.0, 0.0, 1.0);
	v = (1.0/viewer_h.w)*viewer_h.xyz; //< Viewer position
	
	//Set water depth
	h = dz*(c - texture2D(height_map, vec2(gl_MultiTexCoord0.x, gl_MultiTexCoord0.y)).x);
	
	//Set output vertex position
	gl_Position = gl_ModelViewProjectionMatrix*position;
	
	//Set texture coordinates
	gl_TexCoord[0] = gl_MultiTexCoord0;
}