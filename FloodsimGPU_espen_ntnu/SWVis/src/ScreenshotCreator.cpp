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

#include "ScreenshotCreator.h"

#include <iomanip>
#include <fstream>

#include "GLFuncs.h"

using namespace std;

ScreenshotCreator::ScreenshotCreator(std::string basename) {
	//Create directory for screenshots
	time_t foo = time(NULL);
#pragma warning (disable : 4996)
	tm* t = localtime(&foo);
#pragma warning (default : 4996)
	screenshots_dir << basename;
	screenshots_dir << setfill('0');
	screenshots_dir << "_" << (1900+t->tm_year) << "_" << setw(2) << t->tm_mon << "_" << setw(2) << t->tm_mday;
	screenshots_dir << "_" << setw(2) << t->tm_hour << "_" << setw(2) << t->tm_min << "_" << setw(2) << t->tm_sec;
	counter = 0;
}

void ScreenshotCreator::snap() {
	if (counter == 0) {
		boost::filesystem::create_directory(screenshots_dir.str().c_str());
	}
	stringstream filename;
	filename << screenshots_dir.str() << "/";
	filename << setw(5) << setfill('0') << ++counter << ".tga";
	snap(filename.str());
}

void ScreenshotCreator::snap(std::string filename) {
	int windowWidth, windowHeight;
	unsigned long long size;

	windowWidth = glutGet(GLUT_WINDOW_WIDTH);
	windowHeight = glutGet(GLUT_WINDOW_HEIGHT);

	size = static_cast<unsigned long long>(windowWidth)
			* static_cast<unsigned long long>(windowHeight)*4ul;

	windowBuffer.clear();
	windowBuffer.resize(size);
	assert(windowBuffer.size() == size);
	fileBuffer.clear();
    fileBuffer.reserve(size);

	glReadBuffer(GL_BACK);
	glReadPixels(0, 0, windowWidth, windowHeight, GL_RGB, GL_UNSIGNED_BYTE, &windowBuffer[0]);


    // TGA header
	fileBuffer.push_back( 0 );                        // id length
	fileBuffer.push_back( 0 );                        // color map type
	fileBuffer.push_back( 10 );                       // data type code
    fileBuffer.push_back( 0 );                        // colormap origin LSB
    fileBuffer.push_back( 0 );                        // colormap origin MSB
    fileBuffer.push_back( 0 );                        // colormap length LSB
    fileBuffer.push_back( 0 );                        // colormap length MSB
    fileBuffer.push_back( 0 );                        // color map depth
    fileBuffer.push_back( 0 );                        // x origin LSB
    fileBuffer.push_back( 0 );                        // x origin MSB
    fileBuffer.push_back( 0 );                        // y origin LSB
    fileBuffer.push_back( 0 );                        // y origin MSB
    fileBuffer.push_back( windowWidth & 0xff );       // width LSB
    fileBuffer.push_back( (windowWidth>>8) & 0xff );  // width MSB
    fileBuffer.push_back( windowHeight & 0xff );       // height LSB
    fileBuffer.push_back( (windowHeight>>8) & 0xff );  // height MSB
    fileBuffer.push_back( 24 );                       // bits per pixel
    fileBuffer.push_back( 0 );                        // image descriptor

    for (int y=0; y<windowHeight; y++) {
        //encode one scanline
        GLubyte* l = &windowBuffer[3*windowWidth*y];
        GLubyte* r = &l[3*windowWidth];
        while( l < r ) {
            // build one packet
            fileBuffer.push_back( 0 );     // make room for count
            fileBuffer.push_back( l[2] );  // first pixel
            fileBuffer.push_back( l[1] );
            fileBuffer.push_back( l[0] );

            // First, try to build a RLE packet
            GLubyte* c=l+3;
			while ((c<r)
				   && (c-l < 3*128)
				   && (l[0] == c[0])
				   && (l[1] == c[1])
				   && (l[2] == c[2])) {
				c+=3;
			}

            if( c-l > 3 ) { // Something to compress, opt for RLE-packet
                // store repetitions
                fileBuffer[ fileBuffer.size()-4 ] = ((c-l)/3-1) | 128;
                l = c;
            }
            else { // Nothing to compress, make non-RLE-packet

                // search until end of scanline and packet for possible RLE packet
                for( c=l+3; (c<r) &&
                            (c-l < 3*128) &&
                            (!((c[-3] == c[0]) &&
                               (c[-2] == c[1]) &&
                               (c[-1] == c[2])) ); c+=3) {
                    fileBuffer.push_back(c[2]);
                    fileBuffer.push_back(c[1]);
                    fileBuffer.push_back(c[0]);
                }
                // store non-RLE-packet size
                fileBuffer[ fileBuffer.size() - (c-l) -1 ] = (c-l)/3-1;
                l = c;
            }
        }
    }

    ofstream dump(filename.c_str(), ios::out | ios::trunc | ios::binary);
    dump.write(reinterpret_cast<char*>(&fileBuffer[0]), fileBuffer.size());
	dump.close();
}
