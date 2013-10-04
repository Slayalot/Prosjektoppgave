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
uniform float dz; 

varying float h;

void main(void) {
	float n, s, e, w, c; //< north, south, east, west, center
	vec4 position;
	
	//Look up textures
	c = texture2D(water_height_map, vec2(gl_MultiTexCoord0.x   , gl_MultiTexCoord0.y)).x;
	h = dz*(c - texture2D(height_map, vec2(gl_MultiTexCoord0.x, gl_MultiTexCoord0.y)).x); //water depth
	
	//Displace the vertex according to height map
	position = gl_Vertex;
	position.y = c*dz;
	
	//Set output vertex position
	gl_Position = gl_ModelViewProjectionMatrix*position;
	
	//Set texture coordinates
	gl_TexCoord[0] = gl_MultiTexCoord0;
}