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

#ifndef _SIUT_IO_SNARF_HPP
#define _SIUT_IO_SNARF_HPP

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>

#ifdef NVCC
#define CUDAHD __device__ __host__
#else
#define CUDAHD
#endif

#ifdef DEBUG_IO
#define THROW_IO_ERROR
#endif

namespace siut
{
  /** Namespace that contains tools for reading files, dumping files to ostream with line numbers, and adding line numbers to strings.*/
  namespace io_utils
  {

    /** Reads an entire file into a string and returns it.
     * If DEBUG_IO is defined, will throw a runtime_error if there is a problem opening the file.
     */
    inline std::string snarfFile(const std::string &fname)
    {
      std::ifstream file( fname.c_str());
#ifdef THROW_IO_ERROR
      if (!file)
	{
	  std::stringstream s;
	  s << "error opening " << fname << " in " << __FILE__ << " at " << __LINE__ << std::endl;
	  throw std::runtime_error(s.str());
	}
#endif
      return std::string( std::istreambuf_iterator<char>(file),
		     std::istreambuf_iterator<char>());
    }

    /** Copies the string, adding line numbers, and returns the copied string. */
    inline std::string addLineNumbers(const std::string &orig)
    {
      std::stringstream input(orig), ret;
      std::string temp;
      size_t lCounter = 1;

      while(getline(input, temp))
	{
	  ret << lCounter++ << ": " + temp << std::endl;
	}
      return ret.str();

    }

    /** Dumps the string to the specified ostream, adding linenumbers on the way.
     * \param orig the string to process.
     * \param output the ostream to dump the file to.
     */
    inline void dumpSourceWithLineNumbers(const std::string &orig, std::ostream &output = std::cerr)
    {
      std::stringstream input(orig);
      std::string temp;
      size_t lCounter = 1;
      while(getline(input, temp))
	{
	  output << std::setw(3) << lCounter++ << ": ";
	  output << temp << std::endl;
	}
      output.flush();
    }

  }//end IO namespace
}//end siut namespace
#endif
