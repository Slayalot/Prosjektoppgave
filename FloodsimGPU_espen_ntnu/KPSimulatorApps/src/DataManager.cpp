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

#include "DataManager.h"

#include <stdexcept>


void DataManager::intersectionsToCenters(boost::shared_ptr<Field>& in_field) {
	int in_nx = in_field->nx;
	int in_ny = in_field->ny;
	float* data = in_field->data;

	//Don't process, if input is undefined
	if (in_nx*in_ny == 0) return;

	int out_nx = in_nx-1;
	int out_ny = in_ny-1;

	//Must run in serial...
	for (int j=0; j<out_ny; ++j) {
		for (int i=0; i<out_nx; ++i) {
			float sw = data[    j*in_nx+i  ];
			float se = data[    j*in_nx+i+1];
			float nw = data[(j+1)*in_nx+i  ];
			float ne = data[(j+1)*in_nx+i+1];

			data[j*out_nx+i] = 0.25f*(sw+se+nw+ne);
		}
	}

	in_field->nx = out_nx;
	in_field->ny = out_ny;
}

void DataManager::postProcess(boost::shared_ptr<Field>& in_field, DataManager::NO_DATA_VALUE_FILL no_data_value_fill, float dz) {
	float max_value = -std::numeric_limits<float>::max();
	float min_value = std::numeric_limits<float>::max();
	bool no_data_values = false;

	if (dz != 1.0) {
		std::cout << "Warning: scaling all values by dz=" << dz << std::endl;
	}

#pragma omp parallel for
	for (int i = 0; i < static_cast<int>(in_field->nx*in_field->ny); i++) {
		float value = in_field->data[i];
		if (value == in_field->no_data_value) {
			no_data_values = true;
			continue;
		}
		else {
			max_value = std::max(max_value, value);
			min_value = std::min(min_value, value);
			in_field->data[i] = value*dz;
		}
	}

	if (no_data_values) {
		switch(no_data_value_fill) {
		case FILL:
			fillNoDataValues(in_field);
			break;
		case MAX:
			replaceNoDataValues(in_field, max_value);
			break;
		case MIN:
			replaceNoDataValues(in_field, min_value);
			break;
		case ZERO:
			replaceNoDataValues(in_field, 0.0f);
			break;
		default:
			throw std::runtime_error("I don't know what to do with no_data_values!");
			break;
		}
	}
}

void DataManager::replaceNoDataValues( boost::shared_ptr<Field>& in_field, float no_data_value_fill_value ) {
	std::cout << "Warning: replacing no_data_values with " << no_data_value_fill_value << std::endl;
#pragma omp parallel for
	for (int i = 0; i < static_cast<int>(in_field->nx*in_field->ny); i++) {
		float value = in_field->data[i];
		in_field->data[i] = (value == in_field->no_data_value) ? no_data_value_fill_value : value; // set no_data_values to minimal value.
	}
}

void DataManager::fillNoDataValues(boost::shared_ptr<Field>& in_field ) {
	std::cout << "Warning: replacing no_data_values by filling in" << std::endl;
	const float ndv = in_field->no_data_value;
	int no_data_values = 0;

	std::vector<float> values;
	std::vector<int> x;
	std::vector<int> y;

	//Loop through and find the number of no_data_values
#pragma omp parallel for
	for (int i=0; i<static_cast<int>(in_field->nx*in_field->ny); ++i) {
		no_data_values += static_cast<int>(in_field->data[i] == ndv);
	}

	values.reserve(no_data_values);
	x.reserve(no_data_values);
	y.reserve(no_data_values);

	//Then remove them incrementally
	while (no_data_values) {
		//Find the values, x, and y of ndv's
		for (int j = 0; j<static_cast<int>(in_field->ny); j++) {
			for (int i = 0; i < static_cast<int>(in_field->nx); i++) {
				unsigned int jp = std::min(static_cast<int>(in_field->ny)-1, j+1);
				unsigned int jm = std::max(0, j-1);
				unsigned int ip = std::min(static_cast<int>(in_field->nx)-1, i+1);
				unsigned int im = std::max(0, i-1);

				float value = in_field->data[j*in_field->nx + i];
				if (value == ndv) {
					float n = in_field->data[jp*in_field->nx + i];
					float s = in_field->data[jm*in_field->nx + i];
					float e = in_field->data[j*in_field->nx + ip];
					float w = in_field->data[j*in_field->nx + im];

					float sum = 0.0f;
					float count = 0.0f;

					if (n != ndv) {
						sum += n;
						count++;
					}
					if (s != ndv) {
						sum += s;
						count++;
					}
					if (e != ndv) {
						sum += e;
						count++;
					}
					if (w != ndv) {
						sum += w;
						count++;
					}

					if (count > 0) {
						values.push_back(sum/count);
						x.push_back(i);
						y.push_back(j);
					}
				}
			}
		}

		//Fill inn values of the ndv's
#pragma omp parallel for
		for (int i=0; i<static_cast<int>(values.size()); ++i) {
			in_field->data[y.at((i))*in_field->nx + x.at(i)] = values.at(i);
			no_data_values--;
		}

		//Reset contents of the values, x, and y
		values.clear();
		x.clear();
		y.clear();
	}
}
