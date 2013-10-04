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

#ifndef KPBOUNDARYCONDITIONS_H_
#define KPBOUNDARYCONDITIONS_H_

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
#include <limits>
#include <iostream>
#include <vector>
#include <boost/utility.hpp>
#include <cassert>
#include <algorithm>
#include <functional>
#include <cuda_runtime.h>


/**
 * This struct keeps track of everything we need to know about
 * each one boundary condition.
 */
class KPBoundaryCondition {
public:
	enum TYPE {
		NONE=0,            //!< No boundary conditions
		WALL=1,            //!< Wall
		FIXED_DEPTH=2,     //!< Fixed water depth
		FIXED_DISCHARGE=3, //!< Fixed discharge (h*u / h*v)
		OPEN=4,            //!< Open(? = nonreflective?) boundary conditions
		UNKNOWN,           //!< Invalid (unknown) boundary condition
	}; //!< This enum defines the different boundary conditions we can have.

public:
	KPBoundaryCondition(TYPE type_=UNKNOWN) {
		type = type_;
	}

	KPBoundaryCondition(TYPE type_, float value_) {
		type = type_;
		values.push_back(value_);
		times.push_back(0.0f);
	}

	inline std::vector<float>& getValues() { return values; }
	inline const std::vector<float>& getValues() const { return values; }
	inline std::vector<float>& getTimes() { return times; }
	inline const std::vector<float>& getTimes() const { return times; }
	inline TYPE getType() const {return type;}
	
	inline float getValue(float time) const {
		if (type == FIXED_DISCHARGE || type == FIXED_DEPTH) {
			return bilin(times, values, time);
		}
		else {
			return 0.0f;
		}
	}

	inline KPBoundaryCondition& operator=(const KPBoundaryCondition& rhs_) {
		type = rhs_.type;
		values = rhs_.values;
		times = rhs_.times;

		return *this;
	}

	
private:
	static inline float bilin(const std::vector<float>& times, const std::vector<float>& values, float t) {
		float retval;
		unsigned long index;
		for (index=0; index<times.size(); ++index) {
			if (times.at(index) > t) break;
		}
		float last_time = times.at(std::max(1ul, index) - 1);
		float next_time = times.at(std::min(index, static_cast<unsigned long>(times.size()-1)));
		float last_value = values.at(std::max(1ul, index) - 1);
		float next_value = values.at(std::min(index, static_cast<unsigned long>(times.size()-1)));

		if (next_time > last_time) {
			t = (t-last_time) / (next_time-last_time);
			retval = (1.0f-t)*last_value + t*next_value;
		}
		else {
			retval = last_value;
		}
		return retval;
	}

	TYPE type;                 //!< Type of boundary
	std::vector<float> values; //!< Values for boundary condition (if applicable)
	std::vector<float> times;  //!< Times to apply new values (if applicable)
};



/**
 * Helper function to easily print out boundary conditions
 */
inline std::ostream& operator<<(std::ostream& out, const KPBoundaryCondition::TYPE& type) {
	switch (type) {
	case KPBoundaryCondition::NONE:            out << "none"; break;
	case KPBoundaryCondition::WALL:            out << "wall"; break;
	case KPBoundaryCondition::FIXED_DEPTH:     out << "fixed depth"; break;
	case KPBoundaryCondition::FIXED_DISCHARGE: out << "fixed discharge"; break;
	case KPBoundaryCondition::OPEN:            out << "open"; break;
	default:                 out << "unknown: " << (int) type; break;
	}
	return out;
}

inline std::ostream& operator<<(std::ostream& out, const KPBoundaryCondition& bc) {
	out << bc.getType();
	return out;
}









#endif /* KPBOUNDARYCONDITIONS_H_ */
