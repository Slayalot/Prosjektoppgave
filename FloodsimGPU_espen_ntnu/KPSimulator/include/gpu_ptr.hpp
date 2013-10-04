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

#ifndef GPU_PTR_H_
#define GPU_PTR_H_

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#endif

#include <iostream>
#include <limits>
#include <boost/utility.hpp>
#include <boost/shared_ptr.hpp>
#include "gpu_raw_ptr.hpp"
#include "KPException.hpp"

#include <cassert>
#include <vector>
#ifndef NDEBUG
#include <iostream>
#endif

#include <cuda_runtime.h>

#ifndef NDEBUG
namespace {
	static long long total_allocation;
}
#endif

/**
 * This class handles pointer on the GPU, allocation, and deallocation.
 */
template <class T=float>
class gpu_ptr_2D : public boost::noncopyable {
public:
	/**
	 * Allocates 2D data on the GPU, and optionally uploads data
	 * @param width width of data to allocate
	 * @param height height of data to allocate
	 * @param cpu_ptr data to initialize the GPU pointer to (must be width*height elements)
	 */
	gpu_ptr_2D(unsigned int width, unsigned int height, T* cpu_ptr=NULL);

	/**
	 * Deallocates data
	 */
	~gpu_ptr_2D();

	/**
	 * Returns the pointer and pitch to GPU memory allocated by this gpu_ptr. Not
	 * accessible from the CPU directly.
	 * @return pointer in GPU memory space
	 */
	const gpu_raw_ptr<T>& getRawPtr() const {
		return data;
	}

	/**
	 * Returns the width (NOT the pitch) of the allocated memory.
	 * @return width in number of columns
	 */
	const unsigned int& getWidth() const {
		return data_width;
	}

	/**
	 * Returns the height of the allocated memory
	 * @return height in number of rows
	 */
	const unsigned int& getHeight() const {
		return data_height;
	}

	/**
	 * Performs GPU-GPU copy of a width x height domain starting at
	 * x_offset, y_offset from a different gpu_ptr
	 * @param other The gpu_ptr to copy from
	 * @param x_offset The offset to start copying from
	 * @param y_offset The offset to start copying from
	 * @param width Number of elements to copy in x direction
	 * @param height Number of elements to copy in y direction
	 */
	void copy(const gpu_ptr_2D& other,
			unsigned int x_offset=0, unsigned int y_offset=0,
			unsigned int width=0, unsigned int height=0);

	/**
	 * Performs GPU to CPU copy of a width x height domain starting at
	 * x_offset, y_offset from a different gpu_ptr. Best performance is achieved
	 * if data is page locked.
	 * @param cpu_ptr The cpu data to copy to
	 * @param x_offset The offset to start copying from
	 * @param y_offset The offset to start copying from
	 * @param width Number of elements to copy in x direction
	 * @param height Number of elements to copy in y direction
	 */
	void download(T* cpu_ptr,
			unsigned int x_offset=0, unsigned int y_offset=0,
			unsigned int width=0, unsigned int height=0);

	/**
	 * Performs CPU to GPU copy of a width x height domain starting at
	 * x_offset, y_offset from a different gpu_ptr. Best performance is achieved
	 * if data is page locked.
	 * @param cpu_ptr The cpu data to copy from
	 * @param x_offset The offset to start copying from
	 * @param y_offset The offset to start copying from
	 * @param width Number of elements to copy in x direction
	 * @param height Number of elements to copy in y direction
	 */
    void upload(const T* cpu_ptr, unsigned int x_offset=0, unsigned int y_offset=0,
    		unsigned int width=0, unsigned int height=0);

    /**
	 * Performs GPU "memset" of a width x height domain starting at
	 * x_offset, y_offset from a different gpu_ptr. Only really usefull
	 * for setting all bits to 0 or 1.
	 * @param value The bit pattern to set
	 * @param x_offset The offset to start copying from
	 * @param y_offset The offset to start copying from
	 * @param width Number of elements to copy in x direction
	 * @param height Number of elements to copy in y direction
     */
    void set(int value, unsigned int x_offset=0, unsigned int y_offset=0,
			unsigned int width=0, unsigned int height=0);

    /**
	 * Performs GPU "memset" of a width x height domain starting at
	 * x_offset, y_offset from a different gpu_ptr. This is slow to perform, since it 
	 * uses floating point values.
	 * @param value The bit pattern to set
	 * @param x_offset The offset to start copying from
	 * @param y_offset The offset to start copying from
	 * @param width Number of elements to copy in x direction
	 * @param height Number of elements to copy in y direction
     */
    void set(T value, unsigned int x_offset=0, unsigned int y_offset=0,
			unsigned int width=0, unsigned int height=0);

	/**
	 * Swaps to gpu_ptrs.
	 * @param a The first pointer
	 * @param a The second pointer
	 */
	template<class U>
	friend void swap(gpu_ptr_2D<U>& a, gpu_ptr_2D<U>& b);

private:
    gpu_raw_ptr<T> data;
    unsigned int data_width, data_height;
};

template<class T>
inline gpu_ptr_2D<T>::gpu_ptr_2D(unsigned int width, unsigned int height, T* cpu_ptr) {
	data_width = width;
	data_height = height;
	data.ptr = 0;
	data.pitch = 0;
#ifndef NDEBUG
	std::cout << "Allocating [" << data_width << "x" << data_height << "] buffer. " << std::flush;
#endif
	KPSIMULATOR_CHECK_CUDA(cudaMallocPitch((void**) &data.ptr, &data.pitch, data_width*sizeof(T), data_height));
#ifndef NDEBUG
	long long allocated = data.pitch*data_height;
	total_allocation += allocated;
	std::cout << allocated/(1024*1024) << "MB (" << total_allocation/(1024*1024) << "MB) allocated." << std::endl << std::flush;
#endif
	if (cpu_ptr != NULL) upload(cpu_ptr);
}

template<class T>
inline gpu_ptr_2D<T>::~gpu_ptr_2D() {
#ifndef NDEBUG
	std::cout << "Freeing [" << data_width << "x" << data_height << "] buffer. " << std::flush;
	long long allocated = data.pitch*data_height;
#endif
	KPSIMULATOR_CHECK_CUDA(cudaFree(data.ptr));
#ifndef NDEBUG
	total_allocation -= allocated;
	std::cout << allocated/(1024*1024) << "MB (" << total_allocation/(1024*1024) << "MB) freed." << std::endl << std::flush;
#endif
}

template<class T>
inline void gpu_ptr_2D<T>::copy(const gpu_ptr_2D& other, unsigned int x_offset,
		unsigned int y_offset, unsigned int width, unsigned int height) {
	width = (width == 0) ? data_width : width;
	height = (height == 0) ? data_height : height;

	size_t pitch1 = data.pitch;
	T* ptr1 = (T*) ((char*) data.ptr+y_offset*pitch1) + x_offset;

	size_t pitch2 = other.getRawPtr().pitch;
	T* ptr2 = (T*) ((char*) other.getRawPtr().ptr+y_offset*pitch2) + x_offset;

	assert(data_width == other.getWidth()
			&& data_height == other.getHeight()
			&& ptr1 != ptr2);

	KPSIMULATOR_CHECK_CUDA(cudaMemcpy2D(ptr1, pitch1, ptr2, pitch2,
			width*sizeof(T), height, cudaMemcpyDeviceToDevice));
}

template<class T>
inline void gpu_ptr_2D<T>::download(T* cpu_ptr, unsigned int x_offset, unsigned int y_offset,
		unsigned int width, unsigned int height) {
	width = (width == 0) ? data_width : width;
	height = (height == 0) ? data_height : height;

	size_t pitch1 = width*sizeof(T);
	T* ptr1 = cpu_ptr;

	size_t pitch2 = data.pitch;
	T* ptr2 = (T*) ((char*) data.ptr+y_offset*pitch2) + x_offset;

	KPSIMULATOR_CHECK_CUDA(cudaMemcpy2D(ptr1, pitch1, ptr2, pitch2,
			width*sizeof(T), height, cudaMemcpyDeviceToHost));
}

template<class T>
inline void gpu_ptr_2D<T>::upload(const T* cpu_ptr, unsigned int x_offset, unsigned int y_offset,
		unsigned int width, unsigned int height) {
	width = (width == 0) ? data_width : width;
	height = (height == 0) ? data_height : height;

	size_t pitch1 = data.pitch;
	T* ptr1 = (T*) ((char*) data.ptr+y_offset*pitch1) + x_offset;

	size_t pitch2 = width*sizeof(T);
	const T* ptr2 = cpu_ptr;

	KPSIMULATOR_CHECK_CUDA(cudaMemcpy2D(ptr1, pitch1, ptr2, pitch2,
			width*sizeof(T), height, cudaMemcpyHostToDevice));
}

template<class T>
inline void gpu_ptr_2D<T>::set(int value, unsigned int x_offset, unsigned int y_offset,
		unsigned int width, unsigned int height) {
	width = (width == 0) ? data_width : width;
	height = (height == 0) ? data_height : height;

	size_t pitch = data.pitch;
	T* ptr = (T*) ((char*) data.ptr+y_offset*pitch) + x_offset;

	KPSIMULATOR_CHECK_CUDA(cudaMemset2D(ptr, pitch, value, width*sizeof(T), height));
}

template<class T>
inline void gpu_ptr_2D<T>::set(T value, unsigned int x_offset, unsigned int y_offset,
		unsigned int width, unsigned int height) {
	width = (width == 0) ? data_width : width;
	height = (height == 0) ? data_height : height;

	std::vector<T> tmp(width*height, value);

	size_t pitch1 = data.pitch;
	T* ptr1 = (T*) ((char*) data.ptr+y_offset*pitch1) + x_offset;

	size_t pitch2 = width*sizeof(T);
	const T* ptr2 = &tmp[0];

	KPSIMULATOR_CHECK_CUDA(cudaMemcpy2D(ptr1, pitch1, ptr2, pitch2,
		width*sizeof(T), height, cudaMemcpyHostToDevice));
}

template<class T>
inline void swap(gpu_ptr_2D<T>& a, gpu_ptr_2D<T>& b) {
	gpu_raw_ptr<T> tmp = a.data;
	a.data = b.data;
	b.data = tmp;
}

template<class T>
inline std::ostream& operator<<(std::ostream& out, const gpu_ptr_2D<T>& ptr) {
	out << "[" << ptr.getWidth() << "x" << ptr.getHeight() << "]: "
			"ptr=" << ptr.getRawPtr().ptr <<
			" pitch=" << ptr.getRawPtr().pitch;
	return out;
}












/**
 * 1D analogy to gpu_ptr_2D. See documentation of gpu_ptr_2D for use
 */
template<class T=float>
class gpu_ptr_1D : public boost::noncopyable {
public:
	gpu_ptr_1D(unsigned int width, T* cpu_ptr=NULL);
	~gpu_ptr_1D();
	void download(T* cpu_ptr, unsigned int x_offset=0, unsigned int width=0);
	void upload(const T* cpu_ptr, unsigned int x_offset=0, unsigned int width=0);
	void set(int value, unsigned int x_offset=0, unsigned int width=0);
	void set(T value, unsigned int x_offset=0, unsigned int width=0);
	T* getRawPtr() const {
		return data_ptr;
	}
	const unsigned int& getWidth() const {
		return data_width;
	}

private:
	T* data_ptr;
	unsigned int data_width;
};

template<class T>
inline gpu_ptr_1D<T>::gpu_ptr_1D(unsigned int width, T* cpu_ptr) {
	data_width = width;
	data_ptr = 0;
#ifndef NDEBUG
	std::cout << "Allocating [" << data_width << "] buffer. " << std::flush;
#endif
	KPSIMULATOR_CHECK_CUDA(cudaMalloc((void**) &data_ptr, data_width*sizeof(T)));
#ifndef NDEBUG
	long long allocated = data_width*sizeof(T);
	total_allocation += allocated;
	std::cout << allocated/(1024*1024) << "MB (" << total_allocation/(1024*1024) << "MB) allocated." << std::endl << std::flush;
#endif
	if (cpu_ptr != NULL) upload(cpu_ptr);
}

template<class T>
inline gpu_ptr_1D<T>::~gpu_ptr_1D() {
#ifndef NDEBUG
	std::cout << "Freeing [" << data_width << "] buffer. " << std::flush;
	long long allocated = data_width*sizeof(T);
#endif
	KPSIMULATOR_CHECK_CUDA(cudaFree(data_ptr));
#ifndef NDEBUG
	total_allocation -= allocated;
	std::cout << allocated/(1024*1024) << "MB (" << total_allocation/(1024*1024) << "MB) freed." << std::endl << std::flush;
#endif
}

template<class T>
inline void gpu_ptr_1D<T>::download(T* cpu_ptr, unsigned int x_offset, unsigned int width) {
	width = (width == 0) ? data_width : width;

	T* ptr1 = cpu_ptr;
	T* ptr2 = data_ptr + x_offset;

	KPSIMULATOR_CHECK_CUDA(cudaMemcpy(ptr1, ptr2, width*sizeof(T), cudaMemcpyDeviceToHost));
}

template<class T>
inline void gpu_ptr_1D<T>::upload(const T* cpu_ptr, unsigned int x_offset, unsigned int width) {
	width = (width == 0) ? data_width : width;

	T* ptr1 = data_ptr + x_offset;
	const T* ptr2 = cpu_ptr;

	KPSIMULATOR_CHECK_CUDA(cudaMemcpy(ptr1, ptr2, width*sizeof(T), cudaMemcpyHostToDevice));
}

template<class T>
inline void gpu_ptr_1D<T>::set(int value, unsigned int x_offset, unsigned int width) {
	width = (width == 0) ? data_width : width;

	T* ptr = data_ptr + x_offset;

	KPSIMULATOR_CHECK_CUDA(cudaMemset(ptr, value, width*sizeof(T)));
}

template<class T>
inline void gpu_ptr_1D<T>::set(T value, unsigned int x_offset, unsigned int width) {
	width = (width == 0) ? data_width : width;

	T* ptr = data_ptr + x_offset;
	std::vector<T> tmp(width, value);

	KPSIMULATOR_CHECK_CUDA(cudaMemcpy(ptr, &tmp[0], width*sizeof(T), cudaMemcpyHostToDevice));
}

template<class T>
inline std::ostream& operator<<(std::ostream& out, const gpu_ptr_1D<T>& ptr) {
	out << "[" << ptr.getWidth() << "]: "
			"ptr=" << ptr.getRawPtr();
	return out;
}



typedef boost::shared_ptr<gpu_ptr_2D<float> > shared_gpu_ptr_2D_array3[3];
typedef boost::shared_ptr<gpu_ptr_2D<float> > shared_gpu_ptr_2D_array2[2];
typedef boost::shared_ptr<gpu_ptr_2D<float> > shared_gpu_ptr_2D;
typedef boost::shared_ptr<gpu_ptr_1D<float> > shared_gpu_ptr_1D_array3[3];
typedef boost::shared_ptr<gpu_ptr_1D<float> > shared_gpu_ptr_1D;

/**
 * Helper function to easily print out shared_gpu_ptr_2D_array3
 */
inline std::ostream& operator<<(std::ostream& out, const shared_gpu_ptr_2D_array3& array) {
	for(int i=0; i<3; ++i) {
		out << *(array[i].get()) << std::endl;
	}

	return out;
}

#endif /* GPU_PTR_H_ */
