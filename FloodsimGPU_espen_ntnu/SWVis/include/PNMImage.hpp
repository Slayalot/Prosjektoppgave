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

#ifndef PNMIMAGE_H_
#define PNMIMAGE_H_

#include <iostream>
#include <fstream>
#include <exception>
#include <string>

#include <boost/shared_ptr.hpp>
#include <boost/cstdint.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif





namespace util {

class PNMImageExcpetion : public std::exception {
public:
    PNMImageExcpetion(const std::string& text_) {
        text = text_;
    }
    ~PNMImageExcpetion() throw() {}
    
    virtual const char* what() const throw() {
        return text.c_str();        
    }
private:
    std::string text;
};















//Anonymous inline functions
namespace {

inline void stripComment(std::ifstream& file) {
    char c;
    file >> c;
    if (c == '#')
		while (file.get() != '\n');
        //file.ignore(65336, '\n');
    else 
        file.putback(c);
}

/**
  * Reads the header of a PNM-image
  * @templateparam magic Expected magic number.
  * @param file The (binary) ifstream to read from.
  * @param bpc number of *byte* per pixel channel
  * @param max_value maximum value in image
  * @param width Image width in pixels
  * @param height Image height in pixels
  */
template <char magic>
inline void readPNMHeader(std::ifstream& file, 
						  unsigned int& bpc, unsigned int& max_value, 
						  unsigned int& width, unsigned int& height) {
	char m[2];

    //1. "magic number"
    stripComment(file);
    file >> m[0];//P
    file >> m[1];//1..6
    if (m[0]!= 'P' || m[1] != magic)
        throw PNMImageExcpetion("Wrong magic number");
    
    //3-5. Width and height
    stripComment(file);
    file >> width;
    stripComment(file);
    file >> height;
    
    //7 maximum gray value
    stripComment(file);
    file >> max_value;
    if (max_value > 65535) 
        throw PNMImageExcpetion("Wrong max value");
    bpc = (max_value < 256) ? 1 : 2;
    
    //8 skip whitespace
    stripComment(file);
}

/**
  * Reads the header of a PNM-image
  * @templateparam magic Expected magic number.
  * @param file The (binary) ifstream to read from.
  * @param bpc number of *byte* per pixel channel
  * @param max_value maximum value in image
  * @param width Image width in pixels
  * @param height Image height in pixels
  */
template <char magic>
inline void writePNMHeader(std::ofstream& file, 
						   unsigned int& bpc, unsigned int& max_value,
						   unsigned int& width, unsigned int& height) {
	max_value = 250;//65535;
	bpc = 1;

    //1. "magic number"
	file << 'P' << magic << std::endl;

	//comment
	file << "#Written by babrodtks really cool PNMwriter" << std::endl;
    
    //3-5. Width and height
	file << width << " " << height << std::endl;
    
    //7 maximum gray value
    file << max_value << std::endl;
}

inline boost::uint16_t read16BitValue(boost::uint8_t* data) {
	boost::uint16_t pixel;
	pixel  = data[1] << 8;
	pixel |= data[0];
	return pixel;
}

inline void fillBuffer(std::ifstream& file, std::streamsize size, boost::uint8_t*& buffer) {
	std::streamsize read_bytes = 0;
	unsigned int i = 0;
	while (read_bytes != size && ++i<100) {
		file.read((char*) buffer, size);
		read_bytes += file.gcount();
	}

    if (read_bytes != size) 
        throw PNMImageExcpetion("Unable to read pixel data properly");
}

inline void writeBuffer(std::ofstream& file, unsigned int size, boost::uint8_t*& buffer) {
	file.write((char*) buffer, size);

    if (!file.good()) 
        throw PNMImageExcpetion("Unable to write pixel data properly");
}

inline void openIfstream(const char*& filename, std::ifstream& file) {
	file.open(filename, std::ios::in | std::ios::out | std::ios::binary);
	if (!file)
		throw PNMImageExcpetion("Unable to open file");
}

inline void openOfstream(const char*& filename, std::ofstream& file) {
	file.open(filename, std::ios::out | std::ios::binary);
	if (!file)
		throw PNMImageExcpetion("Unable to open file");
}


}













    
/**
 * Reads an image in pgm-format, and returns values in floating point scaled between 0 and 1.
 *
 * From http://netpbm.sourceforge.net/doc/pgm.html:
 * 
 * Each PGM image consists of the following:
 *
 * 1. A "magic number" for identifying the file type. A pgm image's magic 
 *    number is the two characters "P5".
 * 2. Whitespace (blanks, TABs, CRs, LFs).
 * 3. A width, formatted as ASCII characters in decimal.
 * 4. Whitespace.
 * 5. A height, again in ASCII decimal.
 * 6. Whitespace.
 * 7. The maximum gray value (Maxval), again in ASCII decimal. Must be less 
 *    than 65536, and more than zero.
 * 8. A single whitespace character (usually a newline).
 * 9. A raster of Height rows, in order from top to bottom. Each row 
 *    consists of Width gray values, in order from left to right. Each gray 
 *    value is a number from 0 through Maxval, with 0 being black and Maxval
 *    being white. Each gray value is represented in pure binary by either 1
 *    or 2 bytes. If the Maxval is less than 256, it is 1 byte. Otherwise, 
 *    it is 2 bytes. The most significant byte is first.
 *
 * A row of an image is horizontal. A column is vertical. The pixels in the
 * image are square and contiguous.
 *
 * Each gray value is a number proportional to the intensity of the pixel, 
 * [...]
 * Note that a common variation on the PGM format is to have the gray value
 * be "linear," [...]. pnmgamma takes such a PGM variant as input and produces
 * a true PGM as output.
 *
 * Strings starting with "#" may be comments, the same as with PBM. 
 */
template <typename T>
class PGMImage {
public:
    PGMImage(size_t width, size_t height);
    ~PGMImage();
    
    T* getGrayData();
    void setGrayData(T* data);
    size_t getWidth();
    size_t getHeight();

    static boost::shared_ptr<PGMImage> read(const char* filename);
    static void write(boost::shared_ptr<PGMImage>& image, const char* filename);
    //static write(boost::shared_ptr<PGMIMAGE>const char* filename);
    
private:
    PGMImage() {};
    
private:
    T* data;
    size_t width;
    size_t height;
};




template <typename T> inline T* PGMImage<T>::getGrayData() { return data; }
template <typename T> inline void PGMImage<T>::setGrayData(T* data) { this->data = data; }
template <typename T> inline size_t PGMImage<T>::getWidth() { return width; }
template <typename T> inline size_t PGMImage<T>::getHeight() { return height; }

template <typename T>
PGMImage<T>::PGMImage(size_t width, size_t height) {
    data = new T[width*height];
    this->width = width;
    this->height = height;
}

template <typename T>
PGMImage<T>::~PGMImage() {
    delete [] data;    
}

template <typename T>
boost::shared_ptr<PGMImage<T> > PGMImage<T>::read(const char* filename) {    
    boost::shared_ptr<PGMImage<T> > img;

	std::ifstream file;
    T* data;
    boost::uint8_t* buffer;

    unsigned int width;
    unsigned int height;
    unsigned int max_value;
    unsigned int bpc;
    
	//Open file and read header
    openIfstream(filename, file);
	readPNMHeader<'5'>(file, bpc, max_value, width, height);

    //Allocate data and read into buffer
    img.reset(new PGMImage<T>(width, height));
	buffer = new boost::uint8_t[width*height*bpc];
	fillBuffer(file, width*height*bpc, buffer);
    
    //Convert to float representation between 0 and 1
    data = img->getGrayData();
    if (bpc == 2) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1)
#endif
        for(int i=0; i<(int) (width*height); ++i) {
            data[i] = read16BitValue(&buffer[2*i]) / (T) max_value;
        }
    }
    else { 
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1)
#endif
        for(int i=0; i<(int) (width*height); ++i) {
            data[i] = buffer[i] / (T) max_value;
        }
    }
    
	delete [] buffer;
    file.close();
    
    return img;
}

template <typename T>
void PGMImage<T>::write(boost::shared_ptr<PGMImage>& img, const char* filename) {
    std::ofstream file;
    T* data;
    uint8_t* buffer;

    unsigned int width = img->getWidth();
    unsigned int height = img->getHeight();
    unsigned int max_value;
    unsigned int bpc;

	//Open file and read header
	openOfstream(filename, file);
	writePNMHeader<'5'>(file, bpc, max_value, width, height);

    //Allocate data and read into buffer
    buffer = new uint8_t[width*height*bpc];

    //Convert to float representation between 0 and 1
    data = img->getGrayData();
    if (bpc == 2) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1)
#endif
        for(int i=0; i<width*height; ++i) {
				buffer[2*i  ] = data[i]*max_value;
				buffer[2*i+1] = data[i+1]*max_value;
        }
    }
    else {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1)
#endif
        for(int i=0; i<width*height; ++i) {
				buffer[i] = data[i]*max_value;
        }
    }

	writeBuffer(file, width*height*bpc, buffer);

	delete [] buffer;

    file.close();
}















/**
  * Essentially the same as PGMImage, but three channels interleaved. Thus as above except
  * 1. A "magic number" for identifying the file type. A ppm image's magic number is the two
  *    characters "P6". 
  * 9. A raster of Height rows, in order from top to bottom. Each row consists of Width pixels, 
  *    in order from left to right. Each pixel is a triplet of red, green, and blue samples, in 
  *    that order. Each sample is represented in pure binary by either 1 or 2 bytes. If the Maxval
  *    is less than 256, it is 1 byte. Otherwise, it is 2 bytes. The most significant byte is first. 
  */
template <typename T>
class PPMImage {
public:
	PPMImage(size_t width, size_t height);
	~PPMImage();

	T* getRedData();
	T* getGreenData();
	T* getBlueData();
	size_t getWidth();
	size_t getHeight();
	
    static boost::shared_ptr<PPMImage> read(const char* filename);
    static void write(boost::shared_ptr<PPMImage>& image, const char* filename);

private:
	PPMImage() {};

private:
    T* red, * green, * blue;
    size_t width;
    size_t height;
};






template <typename T> inline T* PPMImage<T>::getRedData() { return red; }
template <typename T> inline T* PPMImage<T>::getGreenData() { return green; }
template <typename T> inline T* PPMImage<T>::getBlueData() { return blue; }
template <typename T> inline size_t PPMImage<T>::getWidth() { return width; }
template <typename T> inline size_t PPMImage<T>::getHeight() { return height; }

template <typename T> 
PPMImage<T>::PPMImage(size_t width, size_t height) {
    red = new T[width*height];
    green = new T[width*height];
    blue = new T[width*height];
    this->width = width;
    this->height = height;
}

template <typename T> 
PPMImage<T>::~PPMImage() {
    delete [] red;
    delete [] green;
    delete [] blue;
}

template <typename T> 
boost::shared_ptr<PPMImage<T> > PPMImage<T>::read(const char* filename) {
    boost::shared_ptr<PPMImage<T> > img;

    std::ifstream file;
    T* data[3];
	boost::uint8_t* buffer;

    unsigned int width;
    unsigned int height;
    unsigned int max_value;
    unsigned int bpc;
    
	//Open file and read header
	openIfstream(filename, file);
	readPNMHeader<'6'>(file, bpc, max_value, width, height);

    //Allocate data and read into buffer
    img.reset(new PPMImage(width, height));
    buffer = new boost::uint8_t[3*width*height*bpc];
	fillBuffer(file, 3*width*height*bpc, buffer);
    
    //Convert to float representation between 0 and 1
    data[0] = img->getRedData();
    data[1] = img->getGreenData();
    data[2] = img->getBlueData();
    if (bpc == 2) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1)
#endif
        for(int i=0; i<(int) (width*height); ++i) {
			for (int j=0; j<3; ++j) {
				data[j][i] = read16BitValue(&buffer[6*i+2*j]) / (T) max_value;
			}
        }
    }
    else { 
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1)
#endif
        for(int i=0; i<(int) (width*height); ++i) {
			for (int j=0; j<3; ++j) {
				data[j][i] = buffer[3*i+j] / (T) max_value;
			}
        }
    }
    
	delete [] buffer;
    file.close();
    
    return img;
}


template <typename T> 
void PPMImage<T>::write(boost::shared_ptr<PPMImage>& img, const char* filename) {
    std::ofstream file;
    T* data[3];
    uint8_t* buffer;

    unsigned int width = img->getWidth();
    unsigned int height = img->getHeight();
    unsigned int max_value;
    unsigned int bpc;
    
	//Open file and read header
	openOfstream(filename, file);
	writePNMHeader<'6'>(file, bpc, max_value, width, height);

    //Allocate data and read into buffer
    buffer = new uint8_t[3*width*height*bpc];
    
    //Convert to float representation between 0 and 1
    data[0] = img->getRedData();
    data[1] = img->getGreenData();
    data[2] = img->getBlueData();
    if (bpc == 2) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1)
#endif
        for(int i=0; i<width*height; ++i) {
			for (int j=0; j<3; ++j) {
				buffer[6*i  ] = data[0][i]*max_value;
				buffer[6*i+1] = data[0][i+1]*max_value;
				buffer[6*i+2] = data[1][i]*max_value;
				buffer[6*i+3] = data[1][i+1]*max_value;
				buffer[6*i+4] = data[2][i]*max_value;
				buffer[6*i+5] = data[2][i+1]*max_value;
			}
        }
    }
    else { 
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1)
#endif
        for(int i=0; i<width*height; ++i) {
			for (int j=0; j<3; ++j) {
				buffer[3*i  ] = data[0][i]*max_value;
				buffer[3*i+1] = data[1][i]*max_value;
				buffer[3*i+2] = data[2][i]*max_value;
			}
        }
    }
    
	writeBuffer(file, 3*width*height*bpc, buffer);
	
	delete [] buffer;

    file.close();
}











}

#endif /*PNMIMAGE_H_*/
