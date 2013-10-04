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


uniform samplerCube skybox;
uniform sampler2D landscape_tex;
uniform sampler2D bump_map;
uniform float dz; // scales normals
uniform vec3 L;

varying vec3 N;
varying vec3 v;
varying vec3 p;

void main(void) {
	vec3 E, H, N2, R;
	float Idiff, Ispec, Iamb;
	
	N2 = texture2D(bump_map, gl_TexCoord[0].xy*10).xzy;
	N2.y = 0;
	
	N = normalize(N+0.5*N2);
	//N = vec3(0, 1, 0);
	L = normalize(L); //< Light vector
	E = normalize(v-p); //View vector
	//H = normalize(E+L); //Half vector
	R = reflect(-E, N); //Reflective ray (when hitting water surface)
		
	
	Iamb = 0.1;//0.8;
	Idiff = 0.9 * max(dot(L, N), 0.0);
	Idiff = clamp(Idiff, 0.0, 1.0);
	//Ispec = 0.5 * pow(max(dot(N, H), 0.0), 64.0);
	Ispec = 0.1*texture(skybox,R);//clamp(Ispec, 0.0, 1.0);
	
	gl_FragColor.a = 1;
	gl_FragColor.rgb = (Iamb+Idiff)*texture2D(landscape_tex, gl_TexCoord[0].xy) + Ispec;
}
