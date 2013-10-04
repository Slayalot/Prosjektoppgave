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

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <limits>
#include <omp.h>

#include <boost/shared_ptr.hpp>

#include "FileManager.h"

using std::cout;
using std::endl;
using std::numeric_limits;
using std::max;
using std::min;
using boost::shared_ptr;

int main(int argc, char** argv) {
	float dx;
	int no_data_value;
	shared_ptr<Field> img;
	float min_value, max_value;
	max_value = numeric_limits<float>::min();
	min_value = numeric_limits<float>::max();

	if (argc != 3) {
		cout << "use: " << argv[0] << " <input> <output>" << endl;
		exit(-1);
	}

	img = FileManager::readDEMFile(argv[1]);

#pragma omp parallel for
	for(int i=0; i<img->nx*img->ny; ++i) {
		max_value = max(max_value, img->data[i]);
		min_value = min(min_value, img->data[i]);
	}

	if (min_value != 0.0) {
		cout << "Warning: " << -min_value << " added to all values." << endl;
		cout << "Max = " << max_value << ", min = " << min_value << endl;
		if (max_value-min_value > 255.0) {
			cout << "Values larger than 255 will overflow....." << endl;
		}
	}

#pragma omp parallel for
	for(int i=0; i<img->nx*img->ny; ++i)
		img->data[i] -= min_value;

	FileManager::writePGMFile(argv[2], img);
}
