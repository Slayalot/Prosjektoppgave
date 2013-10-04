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

#ifndef UTIL_HPP_
#define UTIL_HPP_

#include <sstream>
#include <iomanip>

inline std::string printTime(float time) {
	std::stringstream tmp;
	float s=0, m=0, h=0;

	s = time;

	//hours
	while (s >= 60*60) {
		h+=1;
		s -= 60*60;
	}

	//minutes
	while (s >= 60) {
		m+=1;
		s -= 60;
	}

	tmp << std::setfill('0');

	tmp << std::fixed << std::setprecision(0);
	if (time >= 60)	tmp << std::setw(2) << h << ":" << std::setw(2) << m << ":" << std::setw(2) << std::setprecision(0) << s;
	else tmp << std::setprecision(2) << std::setw(6) << std::right << std::setfill(' ') << s << " s";

	return tmp.str();
}

#endif /* UTIL_HPP_ */
