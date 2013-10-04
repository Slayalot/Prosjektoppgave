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


uniform float dz; 
uniform float h_min; 
uniform float h_max;

varying float uv;
varying float h;

 
vec4 HSV2RGB(vec4 hsva) { 
  vec4 rgba; 
  float h = hsva.x, s = hsva.y, v = hsva.z, m, n, f; 
  float i;   
  if( h == 0.0 ) 
    rgba = vec4(v, v, v, hsva.a); 
  else { 
    i = floor(h); 
    f = h - i; 
    float t = i / 2.0; 
    if( t - floor( t ) <  0.1 ) 
      f = 1.0 - f; // if i is even 
    m = v * (1.0 - s); 
    n = v * (1.0 - s * f); 
    if(i == 0.0 )       rgba = vec4(v, n, m, hsva.a); 
    else if( i == 1.0 ) rgba = vec4(n, v, m, hsva.a); 
    else if( i == 2.0 ) rgba = vec4(m, v, n, hsva.a); 
    else if( i == 3.0 ) rgba = vec4(m, n, v, hsva.a); 
    else if( i == 4.0 ) rgba = vec4(n, m, v, hsva.a); 
    else                rgba = vec4(v, m, n, hsva.a); 
  }
  return rgba; 
}

void main(void) {
	vec4 color = 0;
	vec3 E, H, R;
	float shore_depth = dz;
	
	const vec4 hsv1 = vec4(2.09, 1, 1, 1);
	const vec4 hsv2 = vec4(0.001, 1, 1, 1);
	
	float water_weight = min(pow(2*h/shore_depth, 2.0), 1.0);
	
	float t = min((uv-h_min)/(h_max-h_min), 1.0);
	vec4 hsv = (1-t) * hsv1 + t* hsv2;
	
	vec4 rgb = HSV2RGB(hsv);
	rgb.a = 0.5*water_weight;
		
	gl_FragColor = rgb;
}
