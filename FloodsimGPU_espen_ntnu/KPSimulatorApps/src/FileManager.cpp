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

#include "FileManager.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <string>
#include <stdexcept>
#include <boost/cstdint.hpp>
#include <limits>
#include <omp.h>
#include <sys/stat.h>

#ifndef WIN32
#include <unistd.h>
#endif

using boost::shared_ptr;
using boost::uint8_t;
using std::ifstream;
using std::ofstream;
using std::ios;
using std::cout;
using std::endl;



inline void stripComment(ifstream& file) {
    char c;
    file >> c;
    if (c == '#')
		while (file.get() != '\n');
        //file.ignore(65336, '\n');
    else
        file.putback(c);
}

bool FileManager::fileExists(const std::string& filename) {
	struct stat buffer;
	if (stat(filename.c_str(), &buffer) == 0) {
		return (buffer.st_mode | S_IFREG) != 0; //Regular file
	}
	else {
		return false;
	}
}

void FileManager::readBCValues(std::string filename, std::vector<float>& times, std::vector<float>& values) {
	using namespace std;
	ifstream file;
	float time, value;

	file.open(filename.c_str());

	if (!file.good()) {
		stringstream log;
		log << "Unable to open '" << filename << "'" << endl;
		throw std::runtime_error(log.str());
	}
	else
		cout << "Reading boundary conditions from " << filename << endl;

	times.clear();
	values.clear();

	while (file >> time >> value) {
		times.push_back(time);
		values.push_back(value);
	}

	file.close();
}

shared_ptr<Field> FileManager::readPGMFile(const char* filename) {
    shared_ptr<Field> img;
	ifstream file;
    uint8_t* buffer;

    unsigned int width;
    unsigned int height;
    unsigned int max_value;
    unsigned int bpc;
	std::streamsize read_bytes = 0;
	unsigned int read_tries = 0;
	char m[2];
	long size;

	//Open file and read header
	file.open(filename, ios::in | ios::binary);
	if (!file) {
		std::stringstream log;
		log << "Unable to open file " << filename << endl;
		throw std::runtime_error(log.str());
	}

	//Read header
    //1. "magic number"
    stripComment(file);
    file >> m[0];//P
    file >> m[1];//1..6
	if (m[0]!= 'P' || m[1] != '5') {
		std::stringstream log;
        log << "Wrong magic number" << endl;
		throw std::runtime_error(log.str());
	}

    //3-5. Width and height
    stripComment(file);
    file >> width;
    stripComment(file);
    file >> height;

    //7 maximum gray value
    stripComment(file);
    file >> max_value;
    if (max_value > 65535) {
		std::stringstream log;
        log << "Wrong max value" << endl;
        throw std::runtime_error(log.str());
    }
    bpc = (max_value < 256) ? 1 : 2;
	if (bpc > 1) {
		std::stringstream log;
		log << "16-bit images not implemented" << endl;
		throw std::runtime_error(log.str());
    }

    //8 skip whitespace
    stripComment(file);

    //Allocate data and read into buffer
    img.reset(new Field(width, height));
	buffer = new uint8_t[width*height*bpc];

	size = width*height*bpc;
	while (read_bytes != size && ++read_tries<100) {
		file.read((char*) buffer, size);
		read_bytes += file.gcount();
	}

    if (read_bytes != size) {
		std::stringstream log;
		log << "Unable to read pixel data properly" << endl;
		throw std::runtime_error(log.str());
    }

    //Convert to float representation
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for(int i=0; i<(int) (width*height); ++i) {
		img->data[i] = buffer[i];
	}

	delete [] buffer;
    file.close();

    return img;
}

void FileManager::writePGMFile(const char* filename, shared_ptr<Field>& img) {
	ofstream file;

    unsigned int width;
    unsigned int height;
    unsigned int bpc;
    unsigned int magic;
	long size;
	unsigned char* buffer;

	width = img->nx;
	height = img->ny;

	bpc = 1;
	size = width*height*bpc;
	magic = 5;

	buffer = new unsigned char[size];

#pragma omp parallel for
	for(int i=0; i<(int) (width*height); ++i) {
		buffer[i] = (unsigned char) img->data[i];
	}

	//Open file and read header
	if (fileExists(filename)) {
		std::stringstream log;
		log << "File exists, " << filename << ". Aborting" << endl;
		throw std::runtime_error(log.str());
	}

	file.open(filename, ios::out | ios::binary);
	if (!file) {
		std::stringstream log;
		log << "Unable to open file " << filename << endl;
		throw std::runtime_error(log.str());
	}

	//Read header
    //1. "magic number"
	file << 'P' << magic << std::endl;

	//comment
	file << "#Written by babrodtks really cool PNMwriter" << std::endl;

    //3-5. Width and height
    file << width << " " << height << endl;

    //7 maximum gray value
    file << (int) 255 << endl;

	file.write((char*) buffer, size);
	delete [] buffer;

    if (!file.good()) {
		std::stringstream log;
		log << "Unable to write pixel data" << endl;
		throw std::runtime_error(log.str());
    }

    file.close();
}

/**
 * Reads a DEM-file, and sets dx and no_data_value. Returns a field.
 * WARNING: This is not robust at all...
 * @param filename
 * @param dx
 * @param no_data_value
 * @return
 */
shared_ptr<Field> FileManager::readDEMFile(const char* filename) {
    shared_ptr<Field> img;
	ifstream fin;
	unsigned int width, height;
	float dx;
	float xllcorner, yllcorner;
	float no_data_value;
	char buffer[255];

	fin.open(filename);
	if (fin.fail()) {
		std::stringstream log;
		log << "Error while opening input file '" << filename << "'" << endl;
		throw std::runtime_error(log.str());
	}
	fin >> buffer >> width;
	fin >> buffer >> height;
	fin >> buffer >> xllcorner;
	fin >> buffer >> yllcorner;
	fin >> buffer >> dx;
	fin >> buffer >> no_data_value;
	img.reset(new Field(width, height));
	for (unsigned int i = 0; i < width*height; i++)
		fin >> img->data[i];
	fin.close();

	img->dx = dx;
	img->dy = dx;
	img->xllcorner = xllcorner;
	img->yllcorner = yllcorner;
	img->no_data_value = no_data_value;

	return img;
}

void FileManager::writeDEMFile(const char* filename, boost::shared_ptr<Field>& field) {
	ofstream fout;

#ifndef WIN32_
	if (fileExists(filename)) {
		std::stringstream log;
		log << "File exists, " << filename << ". Aborting" << endl;
		throw std::runtime_error(log.str());
	}
#endif

	fout.open(filename);
	if (fout.fail()) {
		std::stringstream log;
		log << "Error while opening output file '" << filename << "'" << endl;
		throw std::runtime_error(log.str());
	}
	fout << "ncols " << field->nx << endl;
	fout << "nrows " << field->ny << endl;
	fout << "xllcorner " << field->xllcorner << endl;
	fout << "yllcorner " << field->yllcorner << endl;
	fout << "cellsize " << field->dx << endl;
	fout << "NODATA_value " << field->no_data_value << endl;
	for (unsigned int i = 0; i < field->nx*field->ny; i++)
		fout << field->data[i] << " ";
	fout.close();
}





boost::shared_ptr<Field> FileManager::readFile(std::string filename, std::string filetype) {
	boost::shared_ptr<Field> data;

	std::cout << "Reading '" << filename << "' (" << filetype << ")." << std::endl;

	if (filetype.compare("dem") == 0) {
		data = FileManager::readDEMFile(filename.c_str());
	}
	else if (filetype.compare("ppm") == 0) {
		data = FileManager::readPGMFile(filename.c_str());
	}
	else {
		std::stringstream log;
		log << "Could not deduce filetype. Try again with dem or ppm." << std::endl;
		throw(std::runtime_error(log.str()));
	}

	return data;
}
