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
using boost::shared_ptr;

int main(int argc, char** argv) {
	int no_data_value;
	float max_value = numeric_limits<float>::min();
	shared_ptr<Field> img;

	if (argc != 4) {
		cout << "use: " << argv[0] << " <input> <dx> <output>" << endl;
		exit(-1);
	}

	img = FileManager::readPGMFile(argv[1]);

#pragma omp parallel for
	for (int i=0; i<img->nx*img->ny; ++i)
		max_value = max(img->data[i], max_value);

	img->no_data_value = (int) (max_value + 1);
	img->dx = atof(argv[2]);

	FileManager::writeDEMFile(argv[3], img);
}
