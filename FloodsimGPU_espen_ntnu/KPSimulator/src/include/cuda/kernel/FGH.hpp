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

#include "cuda/util.h"
#include <float.h>

#define NEW_SIGN
#define USE_1D_INDEX

__constant__ FGHKernelArgs fgh_ctx;
texture<float, 2, cudaReadModeElementType> texD;


/**
  * This function returns the following:
  * -1, a<0
  *  0, a=0
  *  1, a>0
  */
inline __device__ float sign(float& a) {
	/**
	  * The following works by bit hacks. In non-obfuscated code, something like
	  *  float r = ((int&)a & 0x7FFFFFFF)!=0; //set r to one or zero
	  *  (int&)r |= ((int&)a & 0x80000000);   //Copy sign bit of a
	  *  return r;
	  */
#ifndef NEW_SIGN
	return (signed((int&)a & 0x80000000) >> 31 ) | ((int&)a & 0x7FFFFFFF)!=0;
#else
	float r = ((int&)a & 0x7FFFFFFF)!=0;
	return copysignf(r, a);
#endif
}

/**
 * @return min(a, b, c), {a, b, c} > 0
 *         max(a, b, c), {a, b, c} < 0
 *         0           , otherwise
 */
inline __device__ float minmod(float a, float b, float c) {
	return 0.25f
		*sign(a)
		*(sign(a) + sign(b))
		*(sign(b) + sign(c))
		*fminf( fminf(fabsf(a), fabsf(b)), fabsf(c) );
}

inline __device__ float derivative(float& left, float& center, float& right) {
	return minmod(KPSIMULATOR_MINMOD_THETA*(center-left),
			0.5f*(right-left),
			KPSIMULATOR_MINMOD_THETA*(right-center));
}

/**
  * Returns the smallest eigenvalue < 0, or 0.
  */
inline __device__ float minEigenVal(float a, float b) {
	return fminf(fminf(a, b), 0.0f);
}

/**
  * Returns the largest eigenvalue > 0, or 0.
  */
inline __device__ float maxEigenVal(float a, float b) {
	return fmaxf(fmaxf(a, b), 0.0f);
}

/**
  * Computes both the flux and the lambda function for F
  * @param U{1,2,3} input vector U
  * @param b Bottom elevation
  * @param g Gravitational constant
  * @param F{1,2,3} output vector F
  * @param L{1,2} eigenvalues
  */
inline __device__ void fluxAndLambdaFuncF(float& h, float& U2, float& U3,
		const float& g,
		float& F1, float& F2, float& F3,
		float& u, float& c) {
	F1 = 0.0f;
	F2 = 0.0f;
	F3 = 0.0f;

	if (h <= KPSIMULATOR_ZERO_FLUX_EPS) {
		u = 0.0f;
		c = 0.0f;
		U2 = 0.0f;
		U3 = 0.0f;
		return;
	}
	else { //Only if water at all...
		c = sqrtf(g*h);

		u = U2 / (float) h;  //hu/h
		F1 = U2;              //hu
		F2 = U2*u+0.5f*g*h*h; //hu^2+0.5gh^2
		F3 = U3*u;            //hvu
	}
}

/**
  * Computes both the flux and the lambda (eigenvalue) function for G
  */
inline __device__ void fluxAndLambdaFuncG(float& h, float& U2, float& U3,
		const float& g,
		float& G1, float& G2, float& G3,
		float& v, float& c) {
	G1 = 0.0f;
	G2 = 0.0f;
	G3 = 0.0f;

	if (h <= KPSIMULATOR_ZERO_FLUX_EPS) {
		v = 0.0f;
		c = 0.0f;
		U2 = 0.0f;
		U3 = 0.0f;
		return;
	}
	else {//Only if water at all...
		c = sqrtf(g*h);

		v = U3 / (float) h;  //hv/h
		G1 = U3;              //hv
		G2 = U2*v;            //huv
		G3 = U3*v+0.5f*g*h*h; //hv^2+0.5gh^2
	}
}


/**
 * Reads our physical variables from main memory into shared memory
 */
template <unsigned int height, unsigned int width>
__forceinline__ __device__ void readUAndB(float (&U)[3][height][width], float (&B)[height][width],
		unsigned int bx_, unsigned int by_) {
	int bx = bx_ * blockDim.x;
	for (int j=threadIdx.y; j<height; j+=blockDim.y) {
		int by = by_ * blockDim.y + j;
		float* Bm_ptr  = device_address2D(fgh_ctx.Bm.ptr, fgh_ctx.Bm.pitch, bx, by);
		float* U1_ptr = device_address2D(fgh_ctx.U1.ptr, fgh_ctx.U1.pitch, bx, by);
		float* U2_ptr = device_address2D(fgh_ctx.U2.ptr, fgh_ctx.U2.pitch, bx, by);
		float* U3_ptr = device_address2D(fgh_ctx.U3.ptr, fgh_ctx.U3.pitch, bx, by);

		for (int i=threadIdx.x; i<width; i+=blockDim.x) {
			B[j][i]    = Bm_ptr[i];
			U[0][j][i] = U1_ptr[i];
			U[1][j][i] = U2_ptr[i];
			U[2][j][i] = U3_ptr[i];
		}
	}
    __syncthreads();
}








/**
 * Reconstructs the value of B at integration points in x direction
 */
template <unsigned int height, unsigned int width>
__forceinline__ __device__ void reconstructBx(float (&RBx)[height][width],
		float (&Bi)[height][width],
		unsigned int p, unsigned int q) {
	RBx[q][p] = 0.5f*(Bi[q][p]+Bi[q+1][p]); //0.5*(down+up)
}

/**
 * Reconstructs the value of B at integration points in y direction
 */
template <unsigned int height, unsigned int width>
__forceinline__ __device__ void reconstructBy(float (&RBy)[height][width],
		float (&Bi)[height][width],
		unsigned int p, unsigned int q) {
	RBy[q][p] = 0.5f*(Bi[q][p]+Bi[q][p+1]); //0.5*(left+right)
}

/**
 * Reads B at intersections and reconstructs B at the 
 * integration points
 */
template <unsigned int height, unsigned int width>
__forceinline__ __device__ void readAndReconstructB(float (&Bi)[height][width], 
            float (&RBx)[height][width],
            float (&RBy)[height][width],
            unsigned int p, unsigned int q,
            unsigned int bx_, unsigned int by_) {
    int bx = bx_ * blockDim.x;
    for (int j=threadIdx.y; j<height; j+=blockDim.y) {
        int by = by_ * blockDim.y + j;
		float* Bi_ptr  = device_address2D(fgh_ctx.Bi.ptr, fgh_ctx.Bi.pitch, bx, by);

		for (int i=threadIdx.x; i<width; i+=blockDim.x) {
			Bi[j][i] = Bi_ptr[i];
		}
	}
    __syncthreads();
    
	/**
	 * Reconstruct B at the integration points
	 */
	reconstructBx(RBx, Bi, p, q);
	reconstructBy(RBy, Bi, p, q);
	if (threadIdx.y == 0) { //Use one warp to perform the extra reconstructions needed
		reconstructBy(RBy, Bi, p, 1);//second row
		reconstructBy(RBy, Bi, p, height-2);//second last row
		reconstructBy(RBy, Bi, p, height-1);//last row
		if (threadIdx.x < height-4) {
			reconstructBx(RBx, Bi, 1, p);//second column
			reconstructBx(RBx, Bi, width-2, p); //second last column
			reconstructBx(RBx, Bi, width-1, p);//last column
		}
	}
    __syncthreads();
}







/**
 * Reconstructs the slopes of U in x direction
 */
template <unsigned int height, unsigned int width>
__forceinline__ __device__ void reconstructUx(float (&RUx)[height][width],
		float (&U)[height][width],
		unsigned int p, unsigned int q) {
	RUx[q][p] = 0.5f*derivative(U[q][p-1], U[q][p], U[q][p+1]);
}

/**
 * Reconstructs the slopes of U in y direction
 */
template <unsigned int height, unsigned int width>
__forceinline__ __device__ void reconstructUy(float (&RUy)[height][width],
		float (&U)[height][width],
		unsigned int p, unsigned int q) {
	RUy[q][p] = 0.5f*derivative(U[q-1][p], U[q][p], U[q+1][p]);
}



template <unsigned int height, unsigned int width>
__forceinline__ __device__ void reconstructU(float (&U)[3][height][width],
                                            float (&RUx)[3][height][width],
                                            float (&RUy)[3][height][width],
                                            float (&RBx)[height][width],
                                            float (&RBy)[height][width],
                                            unsigned int p, unsigned int q) {
    //Reconstruct the plane for each cell
#pragma unroll
    for (int l=0; l<3; ++l) {
    	reconstructUx(RUx[l], U[l], p, q);
    	reconstructUy(RUy[l], U[l], p, q);
		if (threadIdx.y == 0) { //Use one warp to perform the extra reconstructions needed
			reconstructUy(RUy[l], U[l], p, 1);
			reconstructUy(RUy[l], U[l], p, height-2);
			if (threadIdx.x < height-4) {
				reconstructUx(RUx[l], U[l], 1, p);
				reconstructUx(RUx[l], U[l], width-2, p);
			}
		}
    }
	__syncthreads();
}















/**
 * Calculates the flux across the west boundary for cell (p, q)
 * @return The CFL-condition parameter r for this interface.
 */
template <bool sync, unsigned int height, unsigned int width>
inline __device__ float computeFluxWest(float (&U)[3][height][width],
        float (&B)[height][width],
		float (&RUx)[3][height][width],
		float (&RBx)[height][width],
		unsigned int p, unsigned int q) {
	float am, ap, um, up, cm, cp;
	float FG1m, FG2m, FG3m, FG1p, FG2p, FG3p;
    float U1p, U2p, U3p, U1m, U2m, U3m;

    // U at "plus" integration point (from the right)
    U1p = U[0][q][p] - RUx[0][q][p] - RBx[q][p];
    U2p = U[1][q][p] - RUx[1][q][p];
    U3p = U[2][q][p] - RUx[2][q][p];

    // U at "minus" integration point (from the left)
    U1m = U[0][q][p-1] + RUx[0][q][p-1] - RBx[q][p];
    U2m = U[1][q][p-1] + RUx[1][q][p-1];
    U3m = U[2][q][p-1] + RUx[2][q][p-1];

    //Compute fluxes
    fluxAndLambdaFuncF(U1p, U2p, U3p, fgh_ctx.g, FG1p, FG2p, FG3p, up, cp);
    fluxAndLambdaFuncF(U1m, U2m, U3m, fgh_ctx.g, FG1m, FG2m, FG3m, um, cm);

    //Find the minimal and maximal eigenvalues
    am = minEigenVal(um-cm, up-cp);
    ap = maxEigenVal(um+cm, up+cp);

	//We might have to sync since we read and write to the same shared memory.
	if (sync) __syncthreads();

	//Calculate the flux across the west boundary
	//Write our computed fluxes to shared memory.
	//We use RUx, as we are finished with it, and shared memory is scarce.
	if (fabsf(ap-am) > KPSIMULATOR_FLUX_SLOPE_EPS) { //positive or negative slope
		RUx[0][q][p] = ((ap*FG1m - am*FG1p) + ap*am*(U1p-U1m))/(ap-am);
		RUx[1][q][p] = ((ap*FG2m - am*FG2p) + ap*am*(U2p-U2m))/(ap-am);
		RUx[2][q][p] = ((ap*FG3m - am*FG3p) + ap*am*(U3p-U3m))/(ap-am);

		return 0.25f*fgh_ctx.dx/fmaxf(ap, -am);
	}
	else {
		RUx[0][q][p] = 0.0f;
		RUx[1][q][p] = 0.0f;
		RUx[2][q][p] = 0.0f;
		return FLT_MAX;
	}
}

/**
 * Calculates the flux across the south boundary for cell (p, q)
 * @return The CDL-condition parameter r for this interface.
 */
template <bool sync, unsigned int height, unsigned int width>
inline __device__ float computeFluxSouth(float (&U)[3][height][width],
        float (&B)[height][width],
		float (&RUy)[3][height][width],
		float (&RBy)[height][width],
		unsigned int p, unsigned int q) {
	float am, ap, um, up, cm, cp;
    float FG1m, FG2m, FG3m, FG1p, FG2p, FG3p;
    float U1p, U2p, U3p, U1m, U2m, U3m;

    // U at "plus" integration point (from the right)
    U1p = U[0][q][p] - RUy[0][q][p] - RBy[q][p];
    U2p = U[1][q][p] - RUy[1][q][p];
    U3p = U[2][q][p] - RUy[2][q][p];

    // U at "minus" integration point (from the left)
    U1m = U[0][q-1][p] + RUy[0][q-1][p] - RBy[q][p];
    U2m = U[1][q-1][p] + RUy[1][q-1][p];
    U3m = U[2][q-1][p] + RUy[2][q-1][p];

    //Compute fluxes
	fluxAndLambdaFuncG(U1p, U2p, U3p, fgh_ctx.g, FG1p, FG2p, FG3p, up, cp);
    fluxAndLambdaFuncG(U1m, U2m, U3m, fgh_ctx.g, FG1m, FG2m, FG3m, um, cm);

    //Find the minimal and maximal eigenvalues
    am = minEigenVal(um-cm, up-cp);
    ap = maxEigenVal(um+cm, up+cp);

	//We might have to sync since we read and write to the same shared memory.
	if (sync) __syncthreads();

	//Calculate the flux across the west boundary
	//Write our computed fluxes to shared memory.
	//We use RUy, as we are finished with it, and shared memory is scarce.
	if (fabsf(ap-am) > KPSIMULATOR_FLUX_SLOPE_EPS) { //positive or negative slope
		RUy[0][q][p] = ((ap*FG1m - am*FG1p) + ap*am*(U1p-U1m))/(ap-am);
		RUy[1][q][p] = ((ap*FG2m - am*FG2p) + ap*am*(U2p-U2m))/(ap-am);
		RUy[2][q][p] = ((ap*FG3m - am*FG3p) + ap*am*(U3p-U3m))/(ap-am);

		return 0.25f*fgh_ctx.dy/fmaxf(ap, -am);
	}
	else {
		RUy[0][q][p] = 0.0f;
		RUy[1][q][p] = 0.0f;
		RUy[2][q][p] = 0.0f;
		return FLT_MAX;
	}
}

__forceinline__ __device__ float sourceTerm(float h, float bx) {
	if (h > KPSIMULATOR_ZERO_FLUX_EPS) {
		return 0.5f*fgh_ctx.g*bx*h;
	}
	else {
		return 0.0f;
	}
}

template <unsigned int height, unsigned int width>
__forceinline__ __device__ float sourceTermEast(float (&U)[height][width],
        float (&B)[height][width],
        float (&RUx)[height][width],
        float (&RBx)[height][width],
        unsigned int p, unsigned int q) {
	float bx;
	float h;
	
	h = U[q][p]+RUx[q][p]-RBx[q][p+1];
	bx = RBx[q][p+1]-RBx[q][p];

	return sourceTerm(h, bx);
}

template <unsigned int height, unsigned int width>
__forceinline__ __device__ float sourceTermWest(float (&U)[height][width],
        float (&B)[height][width],
        float (&RUx)[height][width],
        float (&RBx)[height][width],
        unsigned int p, unsigned int q) {
	float bx;
	float h;

	h = U[q][p]-RUx[q][p]-RBx[q][p];
	bx = RBx[q][p+1]-RBx[q][p];
	
	return sourceTerm(h, bx);
}

template <unsigned int height, unsigned int width>
__forceinline__ __device__ float sourceTermNorth(float (&U)[height][width],
        float (&B)[height][width],
        float (&RUy)[height][width],
        float (&RBy)[height][width],
        unsigned int p, unsigned int q) {
	float bx;
	float h;

	h = U[q][p]+RUy[q][p]-RBy[q+1][p];
	bx = RBy[q+1][p]-RBy[q][p];

	return sourceTerm(h, bx);
}

template <unsigned int height, unsigned int width>
__forceinline__ __device__ float sourceTermSouth(float (&U)[height][width],
        float (&B)[height][width],
        float (&RUy)[height][width],
        float (&RBy)[height][width],
        unsigned int p, unsigned int q) {
	float bx;
	float h;

	h = U[q][p]-RUy[q][p]-RBy[q][p];
	bx = RBy[q+1][p]-RBy[q][p];

	return sourceTerm(h, bx);
}



/**
 * This is the flux and source term kernel that computes the average
 * source term and the net flux for each cell.
 *
 * Each thread computes the flux across the south and west cell interface.
 * Thus we have nx+1 x ny+1 threads to calculate the flux across nx+1 x ny+1
 * interfaces (where nx x ny is the output domain covered by this block).
 * However, we then calculate the net flux for each cell within the
 * domain, nx x ny cells. We also calculate the source terms for these cells.
 */
template <unsigned int output_width, unsigned int output_height, unsigned int step>
__global__ void
FGHKernel() {
	const int sm_width = output_width+4;
	const int sm_height = output_height+4;
	const int nthreads = output_width*output_height;

	__shared__ float Bi[sm_height][sm_width];     //!< Bathymetry at intersections
	__shared__ float RBx[sm_height][sm_width];    //!< Bathymetry reconstructed at x intersections (i+0.5,j)
	__shared__ float RBy[sm_height][sm_width];    //!< Bathymetry reconstructed at y intersections (i, j+0.5)
	__shared__ float U[3][sm_height][sm_width];   //!< U = [w, hu, hv]
	__shared__ float RUx[3][sm_height][sm_width]; //!< Reconstructed point values of U at x intersections
	__shared__ float RUy[3][sm_height][sm_width]; //!< Reconstructed point values of U at y intersections

	float r=FLT_MAX; //!< Maximum eigenvalue
	float R1=0.0f;   //!< Net flux in and out of each cell (U1)
    float R2=0.0f;   //!< Net flux in and out of each cell (U2)
    float R3=0.0f;   //!< Net flux in and out of each cell (U3)
    float ST2 = 0.0f; 
    float ST3 = 0.0f;

	int bx = blockIdx.x;
	int by = blockIdx.y;

    //Data indexing variables
	unsigned int p = threadIdx.x+2;
	unsigned int q = threadIdx.y+2;
	int out_x = bx*output_width + threadIdx.x+2;    //<! We skip over the two ghost cells
	int out_y = by*output_height + threadIdx.y+2;


	// First, perform early exit if possible.
	//if(early_exit_test<early_exit>()) return;

    // Read the bathymetry from global memory and reconstruct 
    // the value at the integration points
    readAndReconstructB(Bi, RBx, RBy, p, q, bx, by);

    // Read our physical variables
    readUAndB(U, Bi, bx, by);

    // Reconstruct the slopes of our physical variables
    reconstructU(U, RUx, RUy, RBx, RBy, p, q);

    //Compute the source term for our F-fluxes
    ST2 = sourceTermEast(U[0], Bi, RUx[0], RBx, p, q);
    ST2 += sourceTermWest(U[0], Bi, RUx[0], RBx, p, q);
    __syncthreads();

	/**
	 * Compute the flux across the west interface of the eastmost row
	 * This must be computed before the rest because we store the flux in RUx[p][q]
	 * Let one warp perform the flux across the eastmost interface
	 * Then compute the fluxes for the rest for the cells
	 * We do not need to call synchtreads for the extra calculations, since they are
	 * performed by a single warp. For the rest of the calculations, on the other
	 * hand, we need to synchronize, since we reuse RUx/RUy to store the result
	 * We do not need to store the minimum r from the eastmost row, since this
	 * is taken care of in the neighbouring block. (It will cause errors when 
	 * applied to the padded east and south global boundaries.)
	 */
	if (threadIdx.y == 0 && threadIdx.x < output_height) {
		computeFluxWest<false>(U, Bi, RUx, RBx, output_width+2, p);
    }
	r = min(r, computeFluxWest<true>(U, Bi, RUx, RBx, p, q));
    
	//Compute the source term, because this requires both north and center values
	//it needs to be computed before the last row of v-fluxes
	ST3 = sourceTermNorth(U[0], Bi, RUy[0], RBy, p, q);
    ST3 += sourceTermSouth(U[0], Bi, RUy[0], RBy, p, q);
	__syncthreads();


	//Compute the northmost row of v-fluxes across interfaces
	//This must be computed before the rest because we store the flux in RUy[p][q]
	//Let one warp perform the flux across the eastmost interface
	if (threadIdx.y == 0) {
		computeFluxSouth<false>(U, Bi, RUy, RBy, p, output_height+2);
    }
	r = min(r, computeFluxSouth<true>(U, Bi, RUy, RBy, p, q));
    
    //Write out net fluxes and source terms to memory
    __syncthreads();
    if (out_x < fgh_ctx.nx+2 && out_y < fgh_ctx.ny+2) {
        //Now, compute the flux contributions along the east-west direction
        R1 =  (RUx[0][q][p] - RUx[0][q][p+1])/fgh_ctx.dx
            + (RUy[0][q][p] - RUy[0][q+1][p])/fgh_ctx.dy;
        R2 =  (RUx[1][q][p] - RUx[1][q][p+1] - ST2)/fgh_ctx.dx
            + (RUy[1][q][p] - RUy[1][q+1][p])/fgh_ctx.dy;
        R3 =  (RUx[2][q][p] - RUx[2][q][p+1])/fgh_ctx.dx
            + (RUy[2][q][p] - RUy[2][q+1][p] - ST3)/fgh_ctx.dy;

		device_address2D(fgh_ctx.R1.ptr, fgh_ctx.R1.pitch, out_x, out_y)[0] = R1;
		device_address2D(fgh_ctx.R2.ptr, fgh_ctx.R2.pitch, out_x, out_y)[0] = R2;
		device_address2D(fgh_ctx.R3.ptr, fgh_ctx.R3.pitch, out_x, out_y)[0] = R3;
	}


	//Now, find and write out the maximal eigenvalue in this block
	if (step==0) {
		__syncthreads();
		volatile float* B_volatile = Bi[0];
		p = threadIdx.y*blockDim.x+threadIdx.x; //reuse p for indexing

		//Write the maximum eigenvalues computed by this thread into shared memory
		//Only consider eigenvalues within the internal domain
		r = (out_x < fgh_ctx.nx+2 && out_y < fgh_ctx.ny+2) ? r : FLT_MAX;
		Bi[0][p] = r; 
		__syncthreads();		
		
		//First use all threads to reduce min(1024, nthreads) values into 64 values
		//This first outer test is a compile-time test simply to remove statements if nthreads is less than 512.
		if (nthreads >= 512) {
			//This inner test (p < 512) first checks that the current thread should
			//be active in the reduction from min(1024, nthreads) elements to 512. Makes little sense here, but
			//a lot of sense for the last test where there should only be 64 active threads.
			//The second part of this test ((p+512) < nthreads) removes the threads that would generate an
			//out-of-bounds access to shared memory
			if (p < 512 && (p+512) < nthreads) Bi[0][p] = fminf(Bi[0][p], Bi[0][p + 512]); //min(1024, nthreads)=>512
			__syncthreads();
		}
		if (nthreads >= 256) { 
			if (p < 256 && (p+256) < nthreads) Bi[0][p] = fminf(Bi[0][p], Bi[0][p + 256]); //min(512, nthreads)=>256
			__syncthreads();
		}
		if (nthreads >= 128) {
			if (p < 128 && (p+128) < nthreads) Bi[0][p] = fminf(Bi[0][p], Bi[0][p + 128]); //min(256, nthreads)=>128
			__syncthreads();
		}
		if (nthreads >= 64) {
			if (p < 64 && (p+64) < nthreads) Bi[0][p] = fminf(Bi[0][p], Bi[0][p + 64]); //min(128, nthreads)=>64
			__syncthreads();
		}

		//Let the last warp reduce 64 values into a single value
		//Will generate out-of-bounds errors for nthreads < 64
		if (p < 32) {
			if (nthreads >= 64) B_volatile[p] = fminf(B_volatile[p], B_volatile[p + 32]); //64=>32
			if (nthreads >= 32) B_volatile[p] = fminf(B_volatile[p], B_volatile[p + 16]); //32=>16
			if (nthreads >= 16) B_volatile[p] = fminf(B_volatile[p], B_volatile[p +  8]); //16=>8
			if (nthreads >=  8) B_volatile[p] = fminf(B_volatile[p], B_volatile[p +  4]); //8=>4
			if (nthreads >=  4) B_volatile[p] = fminf(B_volatile[p], B_volatile[p +  2]); //4=>2
			if (nthreads >=  2) B_volatile[p] = fminf(B_volatile[p], B_volatile[p +  1]); //2=>1
		}

		if (threadIdx.y + threadIdx.x == 0) fgh_ctx.L[by*gridDim.x + bx] = B_volatile[0];
	}
}
















template<unsigned int step>
void FGHKernelLauncher(const FGHKernelArgs* h_ctx, const KernelConfiguration& config) {
	if(step > 1) {
		std::cout << "This kernel is only valid for 2-step RK" << std::endl;
		exit(-1);
	}

	//Upload parameters to the GPU
	KPSIMULATOR_CHECK_CUDA(cudaMemcpyToSymbolAsync(fgh_ctx, h_ctx, sizeof(FGHKernelArgs), 0, cudaMemcpyHostToDevice, config.stream));

	//Launch kernel
	cudaFuncSetCacheConfig(FGHKernel<KPSIMULATOR_FLUX_BLOCK_WIDTH, KPSIMULATOR_FLUX_BLOCK_HEIGHT, step>, cudaFuncCachePreferShared);
	FGHKernel<KPSIMULATOR_FLUX_BLOCK_WIDTH, KPSIMULATOR_FLUX_BLOCK_HEIGHT, step><<<config.grid, config.block, 0, config.stream>>>();
	KPSIMULATOR_CHECK_CUDA_ERROR("fluxSourceKernel");
}
