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


#include "configure.h"

#ifdef KPSIMULATORAPPS_USE_NETCDF

#include <iostream>
#include <string>
#include <iomanip>
#include <omp.h>
#include <vector>
#include <cmath>

#include "KPSimulator.h"
#include "app/common.hpp"

#include "FileManager.h"
#include "KPNetCDFFile.h"


using boost::shared_ptr;
using std::cout;
using std::endl;
using std::fixed;
using std::setprecision;

float interpolate(shared_ptr<Field> data, float x, float y);

int main(int argc, char** argv) {
	shared_ptr<KPNetCDFFile> ifile;
	shared_ptr<TimeStep> ts;
	shared_ptr<Field> M;
	shared_ptr<Field> init_B;
	float x0, y0, x1, y1, xn, yn, z, dz;
	float dx, dy, dxn, dyn;
	float ds;
	float T_diff;
	float T;
	unsigned int n, m;
	char* filename;
	float* tmp;

	if (argc != 10) {
		cout << "use: " << argv[0] << " <ncfile> <T> <n> <x0> <y0> <x1> <y1> <z> <dz>" << endl;
		cout << " Uses the rectangle with the points (x0,y0), (x1, y1), (x0,y0)+z*n, (x1, y1)+z*n, where n is the normal of (x0, y0) (x1, y1)." << endl;
		exit(-1);
	}

	filename = argv[1];
	T = atof(argv[2]);
	n = atoi(argv[3]);
	x0 = atof(argv[4]);
	y0 = atof(argv[5]);
	x1 = atof(argv[6]);
	y1 = atof(argv[7]);
	z = atof(argv[8]);
	dz = atof(argv[9]);
	xn = -(y1-y0); //Normal
	yn = (x1-x0);
	m = ceil(sqrt(xn*xn+yn*yn)/dz);
	dxn = dz*xn/sqrt(xn*xn+yn*yn); //normalized
	dyn = dz*yn/sqrt(xn*xn+yn*yn);
	dx = (x1-x0)/(float) (n-1);
	dy = (y1-y0)/(float) (n-1);
	ds = sqrt((x1-x0)*(x1-x0)+(y1-y0)*(y1-y0))/(float) (n-1);
	tmp = new float[m*n];

	ifile.reset(new KPNetCDFFile(filename));
	KPInitialConditions ic = ifile->readInitialConditions(init_B, M, ts);
	
	ifile->readTimeStepTime(ts, T);

	cout << setprecision(10) << fixed;
	cout << "t=," << ts->time << endl;

	//Calculate the data points
	for (int j=0; j<m; ++j) {
		float x_start = x0+j*dxn;
		float y_start = y0+j*dyn;
		for (int i=0; i<n; ++i) {
			float x = x_start + i*dx;
			float y = y_start + i*dy;
			float h = interpolate(ts->U[0], x, y);
			tmp[j*n+i] = h;
		}
	}

	//Print out...
	cout << ",x0,y0";
	for (int i=0; i<m; ++i)
		cout << "," << "h" << i;
	cout << endl;
	cout << "xi,,";
	for (int i=0; i<m; ++i)
		cout << "," << x0+i*dxn;
	cout << endl;
	cout << "yi,,";
	for (int i=0; i<m; ++i)
		cout << "," << y0+i*dyn;


	cout << endl;
	for (int i=0; i<n; ++i) {
		cout << "," << i*dx << "," << i*dy;
		for (int j=0; j<m; ++j)
			cout << "," << tmp[j*n+i];
		cout << endl;
	}
	cout << endl;

	delete [] tmp;
}

inline void clamp(unsigned int& coord, unsigned int minimum, unsigned int maximum) {
	coord = std::max(minimum, std::min(maximum, coord));
}

float interpolate(shared_ptr<Field> data, float x, float y) {
	unsigned int nx = data->nx;
	unsigned int ny = data->ny;
	float i = x / data->dx;
	float j = y / data->dy;

	unsigned int im = floor(i);
	unsigned int ip = im+1;
	float it = i-im; //Fractional part

	unsigned int jm = floor(j);
	unsigned int jp = jm+1;
	float jt = j-jm; //Fractional part

	//Make sure that the coordinates fall within the domain
	clamp(im, 0, nx-1);
	clamp(ip, 0, nx-1);
	clamp(jm, 0, ny-1);
	clamp(jp, 0, ny-1);

	if (im<0 || ip>=data->nx || jm<0 || jp>=data->ny) {
		cout << "Wrong coordinate from " << x << ", " << y << endl;
		exit(-1);
	}

//	cout << x << ":" << im << ", " << ip << ", " << it << endl;
//	cout << y << ":" << jm << ", " << jp << ", " << jt << endl;
	//cout << im << ", " << jm <<",";

	//Get the four corners surrounding our coordinate
	float mm = data->data[jm*nx+im];
	float mp = data->data[jm*nx+ip];
	float pp = data->data[jp*nx+ip];
	float pm = data->data[jp*nx+im];

	//Linear interpolation for minus and plus for the x-direction
	float a1 = (1-it)*mm + it*mp;
	float a2 = (1-it)*pm + it*pp;

	//Linear interpolation between the two above
	float a3 = (1-jt)*a1 + jt*a2;

	return a3;
}

#else

#include <iostream>
#include <stdlib.h>

int main(int argc, char** argv) {
	std::cout << "NetCDF support not enabled in makefile" << std::endl;
	exit(-1);
}

#endif
