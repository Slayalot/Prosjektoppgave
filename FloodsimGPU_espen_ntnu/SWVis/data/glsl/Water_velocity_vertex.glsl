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
uniform sampler2D water_hu_map;
uniform sampler2D water_hv_map;
uniform sampler2D height_map;
uniform sampler2D normal_map;
uniform float dz; 

varying float uv;
varying float h;

void main(void) {
	float B, w, hu, hv, u, v; 
	vec4 position;
	
	//Look up textures
	w = texture2D(water_height_map, vec2(gl_MultiTexCoord0.x   , gl_MultiTexCoord0.y)).x;
	B = texture2D(height_map, vec2(gl_MultiTexCoord0.x, gl_MultiTexCoord0.y)).x;
	h = w-B;
	hu = texture2D(water_hu_map, vec2(gl_MultiTexCoord0.x   , gl_MultiTexCoord0.y)).x;
	hv = texture2D(water_hv_map, vec2(gl_MultiTexCoord0.x   , gl_MultiTexCoord0.y)).x;
	
	u = 0;
	v = 0;
	if (h > 0) {
		u = hu / h;
		v = hv / h;
	}
	
	//Displace the vertex according to height map
	position = gl_Vertex;
	position.y = w*dz;
	
	//Set water depth++
	h = h*dz;
	uv = dz*(abs(u)+abs(v));
	
	//Set output vertex position
	gl_Position = gl_ModelViewProjectionMatrix*position;
	
	//Set texture coordinates
	gl_TexCoord[0] = gl_MultiTexCoord0;
}
