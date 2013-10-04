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
uniform vec3 L;
uniform float dz; 
uniform float du; //Texture coordinates x
uniform float dv; // --"-- y

varying vec3 N_water, N_landscape;
varying vec3 v;
varying vec3 p;
varying float h;

void main(void) {
	vec4 color = 0;
	vec3 E, H, R, K;
	float theta_air, sin_theta_water, theta_water;
	float Idiff, Ispec, Iamb;
	float bottom;
	const float eta1 = 1.3330;
	const float eta2 = 1.000293;
	const float eta = eta1/eta2;
	const float blend = 1.2;
	float shore_depth = 1;

	N_water = normalize(N_water); //< Normal
	N_landscape = normalize(N_landscape);
	L = normalize(L); //< Light vector
	E = normalize(v-p); //View vector
	H = normalize(E+L); //Half vector
	R = reflect(-E, N_water); //Reflective ray (when hitting water surface)
	K = refract(-E, N_water, eta); // refractive ray
	theta_air = abs(acos(dot(R, N_water))); //Angle of incomming view ray and normal
	sin_theta_water = eta*sin(theta_air); //Sine of refracted angle

	//Calculate refraction and reflection
	float negative_reflection_weight = (float) -R.y*(R.y<0); // We do not want to reflect if reflection points downwards
	float reflect_weight = min(pow(sin_theta_water/(1+negative_reflection_weight), 7.0), 1.0); //Degree of reflection/refraction
	float water_weight = min((h)/(shore_depth), 1.0);
	float depth_darkening_weight = min(1.0-(10*shore_depth)/(h+10*shore_depth), 0.95); //degree of darkening as water gets deeper
	float y_angle_weight = min((1-reflect_weight)*(1-E.y), 0.5); //Angle between y-axis and surface -> gets blue the shallower the angle
	
	
	//Calculation refraction coordinate.
	theta_water = asin(sin_theta_water);
	vec2 coord = gl_TexCoord[0].xy+vec2(K.x*du, K.z*dv)*h;
	Iamb = 0.1;//0.8;
	Idiff = 0.9 * max(dot(L, N_landscape), 0.0);
	Idiff = clamp(Idiff, 0.0, 1.0);
	Ispec = 0.1*texture(skybox,R);
	bottom = (Iamb+Idiff)*texture2D(landscape_tex, coord) + Ispec;
	
	//darkening due to water depth
	bottom = depth_darkening_weight*vec3(19/255.0,63/255.0,10/255.0) + (1-depth_darkening_weight)*bottom;
	
	//Reflection and refraction
	color.rgb = reflect_weight*0.8*texture(skybox,R) + (1-reflect_weight)*bottom;
	
	//make water blue, if angle between view vector and y is shallow
	color.rgb = (y_angle_weight)*vec3(19/255.0,63/255.0,100/255.0) + (1-y_angle_weight)*color.rgb;
	
	color.a=water_weight;
	
	gl_FragColor =color;
}
