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

#ifndef KPNETCDFFILE_H_
#define KPNETCDFFILE_H_

#include "datatypes.h"

#include <boost/utility.hpp>
#include <boost/shared_ptr.hpp>

#include <netcdfcpp.h>
#ifdef KPSIMULATORAPPS_USE_NETCDF
#include <netcdf.h>
#endif

class KPNetCDFFile : boost::noncopyable {
public:
	KPNetCDFFile(std::string filename);
	~KPNetCDFFile();

	KPInitialConditions readInitialConditions(boost::shared_ptr<Field>& init_B,
			boost::shared_ptr<Field>& init_M,
			boost::shared_ptr<TimeStep>& ts);
	void writeInitialConditions(const KPInitialConditions& init);

	void writeTimeStep(boost::shared_ptr<TimeStep> ts, int index=-1);
	void readTimeStepIndex(boost::shared_ptr<TimeStep> ts, int index=-1);
	void readTimeStepTime(boost::shared_ptr<TimeStep> ts, float time);

	unsigned int getNt();

private:
	void writeDims(const KPInitialConditions& init);
	void writeAtts(const KPInitialConditions& init);
	void writeVars(const KPInitialConditions& init);

	void readDims();
	void readAtts();
	void readVars();
	
private:
	boost::shared_ptr<NcFile> file;

	struct {
		struct {
			NcDim* i;       //!< number of grid cell intersections
			NcDim* j;       //!< number of grid cell intersections
			NcDim* x;       //!< number of grid cells
			NcDim* y;       //!< number of grid cells
			NcDim* t;       //!< Time
		} dims;

		struct {
			NcVar* init_B;   //!< Initial bathymetry
			NcVar* init_M;   //!< Initial manning coefficient
			NcVar* U1;       //!< U1 = w
			NcVar* U2;       //!< U2 = hu
			NcVar* U3;       //!< U3 = hv
			NcVar* i;        //!< i
			NcVar* j;        //!< j
			NcVar* x;        //!< x
			NcVar* y;        //!< y
			NcVar* t;        //!< time
		} vars;

		struct {
			NcAtt* i;
			NcAtt* j;
			NcAtt* x;
			NcAtt* y;
			NcAtt* t;
		} atts;
	} layout;
	long nt;
	long t_index;
	float time_offset;
};

#endif /* KPNETCDFFILE_H_ */
