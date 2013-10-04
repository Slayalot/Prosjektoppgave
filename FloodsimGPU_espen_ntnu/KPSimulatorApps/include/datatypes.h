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

#ifndef DATATYPES_H_
#define DATATYPES_H_

#ifdef WIN32
#define NOMINMAX
#include <windows.h>
#endif
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <boost/utility.hpp>
#include <boost/shared_ptr.hpp>
#include <limits>

#include "KPInitialConditions.h"

/**
 * Very simple class to simply hold the data we need to represent a heightmap
 */
struct Field {
	std::vector<float> store;
	float* data;
	unsigned int nx, ny;
	float dx, dy;
	float xllcorner, yllcorner;
	float no_data_value;

	Field() {
		nx = 0;
		ny = 0;
		no_data_value = std::numeric_limits<float>::max();
		dx = -1;
		dy = -1;
		xllcorner = 0.0f;
		yllcorner = 0.0f;
		data = NULL;
	}
	Field(unsigned int width, unsigned int height) {
		nx = width;
		ny = height;
		store.resize(nx*ny);
		data = &store[0];
		no_data_value = std::numeric_limits<float>::max();
		dx = -1;
		dy = -1;
		xllcorner = 0.0f;
		yllcorner = 0.0f;
	}
	Field(const Field& other) {
		nx = other.nx;
		ny = other.ny;
		dx = other.dx;
		dy = other.dy;
		xllcorner = other.xllcorner;
		yllcorner = other.yllcorner;
		no_data_value = other.no_data_value;
		store.resize(nx*ny);
		data = &store[0];
		std::copy(other.store.begin(), other.store.end(), store.begin());
	}
	~Field() {
		data = NULL;
	}
};

typedef boost::shared_ptr<Field> shared_field_ptr_array3[3];
typedef boost::shared_ptr<Field> shared_field_ptr;

inline std::ostream& operator<<(std::ostream& out, const Field& f) {
	out << "Field [" << f.nx << "x" << f.ny << "]" << "x[" << f.dx << "x" << f.dy << "], no_data_value=" << f.no_data_value << std::endl;

	for (unsigned int i=0; i<f.ny; ++i) {
		for (unsigned int j=0; j<f.nx; ++j) {
			out << std::fixed << std::setprecision(3) << std::setw(6) << f.data[f.nx*i+j] << " ";
		}
		if (i < f.ny-1)
			out << std::endl;
	}
	out << std::setprecision(-1);
	return out;
}

inline std::ostream& operator<<(std::ostream& out, const Field* f) {
	out << *f;
	return out;
}

/**
 * Struct that describes a iteration/timestep, e.g., from a previous simulation.
 */
class TimeStep : boost::noncopyable {
public:
	TimeStep(unsigned int nx, unsigned int ny) {
		U[0].reset(new Field(nx, ny));
		U[1].reset(new Field(nx, ny));
		U[2].reset(new Field(nx, ny));
		this->nx = nx;
		this->ny = ny;
	}

	/**
	 * Water elevation (w) and discharges (hu, hv) given at grid cell
	 * corners and grid cell interfaces, respectively.
	 */
	shared_field_ptr_array3 U;

	/**
	 * Number of nodes for water elevation and discharges
	 * The simulator then creates a simulation domain of nx-1 x ny-1 cells.
	 * The simulator further uses two ghost cells, making the actual internal
	 * simulation domain nx-3 x ny-3 cells.
	 */
	unsigned int nx, ny;

	/**
	 * Time associated with this iteration/timestep.
	 */
	float time;
};

#endif /* DATATYPES_H_ */
