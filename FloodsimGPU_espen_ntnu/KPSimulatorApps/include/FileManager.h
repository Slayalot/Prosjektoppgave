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

#ifndef FILEMANAGER_H_
#define FILEMANAGER_H_

#include <iostream>
#include <vector>
#include <boost/utility.hpp>
#include <boost/shared_ptr.hpp>

#include "datatypes.h"

class KPSource;

/**
 * Struct that reads into initialconditions
 */
class FileManager : boost::noncopyable {
public:
	FileManager() {};

	static boost::shared_ptr<Field> readFile(std::string filename, std::string filetype);
	static boost::shared_ptr<Field> readPGMFile(const char* filename);
	static boost::shared_ptr<Field> readDEMFile(const char* filename);

	static void readBCValues(std::string filename, std::vector<float>& times, std::vector<float>& values);
	static KPSource readSource(std::string filename);

	static void writeDEMFile(const char* filename, boost::shared_ptr<Field>& img);
	static void writePGMFile(const char* filename, boost::shared_ptr<Field>& img);

	static bool fileExists(const std::string& filename);
};


#endif /* FILEHANDLER_H_ */
