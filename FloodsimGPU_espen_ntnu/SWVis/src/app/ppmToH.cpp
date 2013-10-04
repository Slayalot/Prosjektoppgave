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

#include "PNMImage.hpp"

#include <iostream>
#include <string>
#include <iomanip>
#include <cstring>
#include <fstream>



using namespace util;
using std::ofstream;
using std::cout;
using std::endl;
using boost::shared_ptr;

int main(int argc, char* argv[]) {
	char* input_file;
	char* output_file;
	ofstream output;
	shared_ptr<PPMImage<float> > image;
	unsigned int width, height;

	if (argc != 3) {
		cout << "Usage: " << argv[0] << " <shaderfile> <outputfile>" << endl;
		exit(-1);
	}
	else {
		input_file = argv[1];
		output_file = argv[2];
		if (strcmp(input_file, output_file) == 0) {
			cout << "Cannot use the same input and output file." << endl;
			cout << "Input was " << input_file << ", " << endl;
			cout << "Output was " << output_file << "." << endl;
			exit(-1);
		}
		else {
			cout << "Reading from " << input_file << ", writing to " << output_file << "." << endl;
		}
	}

	image = PPMImage<float>::read(input_file);
	width = image->getWidth();
	height = image->getHeight();

	output.open(output_file);
	if (!output.good()) {
		cout << "Error opening " << output_file << " for writing." << endl;
		exit(-1);
	}

	output << "const unsigned int width = " << width << ";" << endl;
	output << "const unsigned int height = " << height << ";" << endl;
	output << "const unsigned char pixel_data[" << height << "][" << width*3 << "]=" << endl;
	output << "{" << endl;
	for (unsigned int j=0; j<height; ++j) {
		output << "{";
		for (unsigned int i=0; i<width; ++i) {
			float r = image->getRedData()[j*width+i]*255.0f;
			float g = image->getGreenData()[j*width+i]*255.0f;
			float b = image->getBlueData()[j*width+i]*255.0f;

			output << (unsigned int) r << ",";
			output << (unsigned int) g << ",";
			output << (unsigned int) b << ",";
			//cout << (unsigned char) r << (unsigned char) g << (unsigned char) b;
		}
		output << "}," << endl;
	}
	output << "};" << endl;

	return 0;
}

