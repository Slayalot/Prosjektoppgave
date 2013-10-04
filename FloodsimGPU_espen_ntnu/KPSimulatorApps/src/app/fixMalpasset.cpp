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
#include <cmath>

#include "FileManager.h"

using std::cout;
using std::endl;
using std::numeric_limits;
using std::min;
using boost::shared_ptr;

int main(int argc, char** argv) {
	int no_data_value;
	float max_value = numeric_limits<float>::min();
	shared_ptr<Field> pgm;
	shared_ptr<Field> dem;
	shared_ptr<Field> tmp;

	if (argc != 4) {
		cout << "use: " << argv[0] << " <input-pgm> <malpasset-dem> <output-dem>" << endl;
		exit(-1);
	}

	pgm = FileManager::readPGMFile(argv[1]);
	dem = FileManager::readDEMFile(argv[2]);
	tmp.reset(new Field(dem->nx, dem->ny));

	for (int j=0; j<dem->ny; ++j) {
		for (int i=0; i<dem->nx; ++i) {
			int c = j*dem->nx+i;
			pgm->data[c] -= 20.0;
		}
	}

	for (int j=0; j<dem->ny; ++j) {
		for (int i=0; i<dem->nx; ++i) {
			int c = j*dem->nx+i;
			if (0.01 > fabs(dem->data[c] - dem->no_data_value) || (dem->data[c] == -20 && pgm->data[c] >= -20)) {
				dem->data[c] = pgm->data[c];
				tmp->data[c] = 1;
			}
			else {
				tmp->data[c] = 0;
			}
		}
	}

	//Center of domain)
	for (int k=0; k<2; ++k) {
	for (int j=1; j<dem->ny-1; ++j) {
		for (int i=1; i<dem->nx-1; ++i) {
			float nw = dem->data[(j+1)*dem->nx+i+1];
			float n  = dem->data[(j+1)*dem->nx+i  ];
			float ne = dem->data[(j+1)*dem->nx+i+1];
			float w  = dem->data[(j  )*dem->nx+i-1];
			float c  = dem->data[(j  )*dem->nx+i  ];
			float e  = dem->data[(j  )*dem->nx+i+1];
			float sw = dem->data[(j-1)*dem->nx+i+1];
			float s  = dem->data[(j-1)*dem->nx+i  ];
			float se = dem->data[(j-1)*dem->nx+i+1];

			float c_pgm = pgm->data[(j  )*dem->nx+i  ];

			//Gaussian blur
			float avg = 1/16.0*(nw+2*n+ne+2*w+2*c_pgm+2*c+2*e+sw+2*s+se);

			if (tmp->data[j*dem->nx+i] == 1)
				//for (int l=0; l<10; ++l)
				dem->data[j*dem->nx+i] = avg;
		}
	}

	//Borders
	for (int i=1; i<dem->nx-1; ++i) {
		int j = 0;
		float w  = dem->data[j*dem->nx+i-1];
		float c  = dem->data[j*dem->nx+i  ];
		float e  = dem->data[j*dem->nx+i+1];

		float c_pgm = pgm->data[j*dem->nx+i];

		//Gaussian blur
		float avg = 1/4.0*(w+c+c_pgm+e);

		if (tmp->data[j*dem->nx+i] == 1)
			//for (int l=0; l<10; ++l)
			dem->data[j*dem->nx+i] = avg;
	}
	for (int i=1; i<dem->nx-1; ++i) {
		int j = dem->ny-1;
		float w  = dem->data[j*dem->nx+i-1];
		float c  = dem->data[j*dem->nx+i  ];
		float e  = dem->data[j*dem->nx+i+1];

		float c_pgm = pgm->data[j*dem->nx+i];

		//Gaussian blur
		float avg = 1/4.0*(w+c+c_pgm+e);

		if (tmp->data[j*dem->nx+i] == 1)
			//for (int l=0; l<10; ++l)
			dem->data[j*dem->nx+i] = avg;
	}
	for (int j=1; j<dem->ny-1; ++j) {
		int i = 0;
		float n  = dem->data[(j+1)*dem->nx+i  ];
		float c  = dem->data[(j  )*dem->nx+i  ];
		float s  = dem->data[(j-1)*dem->nx+i  ];

		float c_pgm = pgm->data[(j  )*dem->nx+i  ];

		//Gaussian blur
		float avg = 1/4.0*(n+c+c_pgm+s);

		if (tmp->data[j*dem->nx+i] == 1)
			//for (int l=0; l<10; ++l)
			dem->data[j*dem->nx+i] = avg;
	}
	for (int j=1; j<dem->ny-1; ++j) {
		int i = dem->nx-1;
		float n  = dem->data[(j+1)*dem->nx+i  ];
		float c  = dem->data[(j  )*dem->nx+i  ];
		float s  = dem->data[(j-1)*dem->nx+i  ];

		float c_pgm = pgm->data[(j  )*dem->nx+i  ];

		//Gaussian blur
		float avg = 1/4.0*(n+c_pgm+c+s);

		if (tmp->data[j*dem->nx+i] == 1)
			//for (int l=0; l<10; ++l)
			dem->data[j*dem->nx+i] = avg;
	}
	}


	FileManager::writeDEMFile(argv[3], dem);
}
