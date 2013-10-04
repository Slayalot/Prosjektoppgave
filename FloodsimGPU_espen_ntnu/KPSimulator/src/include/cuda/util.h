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

#ifndef CUDA_UTIL_H_
#define CUDA_UTIL_H_

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#endif

/**
 * Fix to get Eclipse to parse cuda code correctly...
 */
#ifdef __CDT_PARSER__
#include <cuda_runtime.h>
#define __host__
#define __global__
#define __device__
#define __shared__
#define __constant__
#define __launch_bounds__(foo, bar)
#endif

/**
  * Calculates the address based on input parameters
  */
template <typename T>
__host__ inline T* address2D(T* base, unsigned int pitch, unsigned int x, unsigned int y) {
	return (T*) ((char*) base+y*pitch) + x;
}

/**
  * Calculates the address based on input parameters
  * @param base The start pointer of the array
  * @param pitch number of bytes in each row
  * @param x offset in elements to x direction
  * @param y offset in elements in y direction
  */
template <typename T>
__device__ inline T* device_address2D(T* base, unsigned int pitch, unsigned int x, unsigned int y) {
	return (T*) ((char*) base+y*pitch) + x;
}

#endif /* UTIL_H_ */
