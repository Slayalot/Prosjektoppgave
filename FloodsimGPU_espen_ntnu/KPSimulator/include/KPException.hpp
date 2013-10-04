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

#ifndef KPEXCEPTION_HPP__
#define KPEXCEPTION_HPP__

#include <stdexcept>
#include "cuda.h"
#include "driver_types.h"

/**
  * The exception class that makes it easier to handle errors within
  * the GPUVideo code.
  */
class KPException : public std::runtime_error {
public:
	/**
	  * Constructor
	  */
	KPException(const std::string& err) : std::runtime_error(err) {	}
	
	/**
	  * Helper function to throw an exception with file and line information. 
	  * Will only print out file and line in debug mode
	  */
	static void throwException(const std::string& err, const char* file, unsigned int line) {
#ifndef NDEBUG
		std::stringstream ss;
		ss << "Fatal error in " << file << ":" << line << ":" << std::endl << err;
		throw KPException(ss.str());
#else
		throw KPException(err);
#endif
	}
	
	/**
	  * Helper function that throws an exception if there is a cuda driver error.
	  * Will only print out file and line in debug mode
	  */
	static void throwIfCUDAError(CUresult error, const std::string& msg, const char* file, unsigned int line) {
		if (error != CUDA_SUCCESS) {
			std::stringstream ss;
			ss << "Fatal CUDA error (errno=" << error << "):" << std::endl << msg;
			throwException(ss.str(), file, line);
		}
	}
	
	/**
	  * Helper function that throws an exception if there is a cuda runtime error.
	  * Will only print out file and line in debug mode
	  */
	static void throwIfCUDAError(cudaError error, const std::string& msg, const char* file, unsigned int line) {
		if (error != cudaSuccess) {
			std::stringstream ss;
			ss << "Fatal CUDA error (errno=" << error << "):" << std::endl << msg;
			throwException(ss.str(), file, line);
		}
	}
};

#define KPSIMULATOR_CHECK_CUDA(call) \
	KPException::throwIfCUDAError(call, "Offending call was '" #call "'", __FILE__, __LINE__)
#define KPSIMULATOR_CHECK_CUDA_ERROR(msg) \
	KPException::throwIfCUDAError(cudaGetLastError(), msg, __FILE__, __LINE__)

#endif //include guard