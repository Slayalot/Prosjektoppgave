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

#ifndef DATA_GENERATION_HPP_
#define DATA_GENERATION_HPP_

#include <sstream>
#include <stdexcept>
#include <cmath>

const unsigned int EU_CADAM = 3;
const unsigned int STEADY_FLOW_OVER_BUMP = 4;
const unsigned int IDEALISED_CIRCULAR_DAM = 6;
const unsigned int CADAM_TEST_1 = 7;
const unsigned int PARABOLIC_BASIN = 8;
const unsigned int MGPU_TEST = 9;

namespace parabolic_basin {
//FIXME: Mongo ugly to have these variables as globals...
const float D0 = 1.0f;
const float L = 2500.0f;
const float A = L/2.0f;
const float B = -A/(2.0f*L);
const float omega = sqrt(2.0f*D0/(L*L));
const float t = 0.0f*6.28f/omega;
}

inline boost::shared_ptr<Field> generate_bathymetry(int no, int nx_, int ny_) {
	boost::shared_ptr<Field> f;

	if (nx_ <= 0 || ny_ <= 0) {
		std::stringstream log;
		log << "Invalid nx or ny: [" << nx_ << ", " << ny_ << "]." << std::endl;
		throw std::runtime_error(log.str());
	}

	std::cout << "Generating bathymetry: '";

	f.reset(new Field(nx_+1, ny_+1));

	switch (no) {
	case 0:
		std::cout << "flat";
		for (unsigned int i=0; i<f->nx*f->ny; ++i)
			f->data[i] = 0.0f;
		break;
	case 1:
		std::cout << "peaks";
#pragma omp parallel for
		for (int j=0; j<static_cast<int>(f->ny); ++j) {
			float y = j * 6.0f/(float) ny_ - 3.0f;
			for (unsigned int i=0; i<f->nx; ++i) {
				float x = i * 6.0f/(float) nx_ - 3.0f;
				float value = 3.0f*(1-x)*(1-x) * exp(-(x*x) - (y-1)*(y-1))
								- 10.0f * (x/5.0f - x*x*x - y*y*y*y*y) * exp(-(x*x) - (y*y))
								- 1.0f/3.0f * exp(-(x+1)*(x+1) - (y*y));

				f->data[j*f->nx+i] = 0.1f*value;
			}
		}
		break;
	case 2:
		std::cout << "3 bumps";
#pragma omp parallel for
		for (int j=0; j<static_cast<int>(f->ny); ++j) {
			float y = j / (float) ny_;
			for (unsigned int i=0; i<f->nx; ++i) {
				float x = i / (float) nx_;
				if ((x-0.25f)*(x-0.25f)+(y-0.25f)*(y-0.25f)<0.01)
					f->data[j*f->nx+i] = 5.0f*(0.01f-(x-0.25f)*(x-0.25f)-(y-0.25f)*(y-0.25f));
				else if ((x-0.75f)*(x-0.75f)+(y-0.25f)*(y-0.25f)<0.01f)
					f->data[j*f->nx+i] = 5.0f*(0.01f-(x-0.75f)*(x-0.75f)-(y-0.25f)*(y-0.25f));
				else if ((x-0.25f)*(x-0.25f)+(y-0.75f)*(y-0.75f)<0.01f)
					f->data[j*f->nx+i] = 5.0f*(0.01f-(x-0.25f)*(x-0.25f)-(y-0.75f)*(y-0.75f));
				else
					f->data[j*f->nx+i] = 0.0f;
			}
		}
		break;
	case EU_CADAM:
		std::cout << "EU CADAM";
		f->dx = 38.0f / (float) nx_;
		f->dy = 20.0f / (float) ny_;
#pragma omp parallel for
		for (int j=0; j<static_cast<int>(f->ny); ++j) {
			for (unsigned int i = 0; i<f->nx; ++i) {
				float x = i*f->dx;
				if (x <= 25.5f) {
					f->data[j*f->nx+i] = 0.0f;
				}
				else if (x <= 28.5) {
					f->data[j*f->nx+i] =        0.4f*(x-25.5f)/3.0f;
				}
				else if (x <= 31.5) {
					f->data[j*f->nx+i] = 0.4f - 0.4f*(x-28.5f)/3.0f;
				}
				else {
					f->data[j*f->nx+i] = 0.0f;
				}
			}
		}
		break;
	case STEADY_FLOW_OVER_BUMP:
		std::cout << "Steady Flow Over Bump";
		f->dx = 25.0f / (float) nx_;
		f->dy = 20.0f / (float) ny_;
#pragma omp parallel for
		for (int j=0; j<static_cast<int>(f->ny); ++j) {
			for (unsigned int i = 0; i<f->nx; ++i) {
				float x = i*f->dx;
				if (8.0f < x && x < 12.0f) {
					f->data[j*f->nx+i] = 0.2f - 0.05f*(x-10.0f)*(x-10.0f);
				}
				else {
					f->data[j*f->nx+i] = 0.0f;
				}
			}
		}
		break;
	case IDEALISED_CIRCULAR_DAM:
		f->dx = 100.0f / (float) nx_;
		f->dy = 100.0f / (float) ny_;
		std::cout << "idealized circular dam";
		for (unsigned int i=0; i<f->nx*f->ny; ++i)
			f->data[i] = 0.0f;
		break;
	case CADAM_TEST_1:
		f->dx = 9.575f / (float) nx_;
		f->dy = 3.73f / (float) ny_;
		std::cout << "CADAM Test 1";
		
		// set bathymetry
#pragma omp parallel for
		for (int i=0; i<static_cast<int>(f->nx*f->ny); ++i)
			f->data[i] = 0.70f;
		
#pragma omp parallel for
		for (int j = 0; j < static_cast<int>(f->ny); ++j) {
			float y = f->dy * j;
			for (unsigned int i = 0; i < f->nx; ++i) {
				float x = f->dx * i;
				
				// carve out reservoir
				if(x < 2.39 && y < 2.44) {
					f->data[j*f->nx+i] = 0.0f;
				}
				// carve out canal
				// straight part
				else if (x > 2.39 && y > 0.445 && x < 6.435  && y < 0.94) {
					f->data[j*f->nx+i] = 0.33f;
				}
				// 45 deg part
				else if (y > x-6.195 && y < x-5.495 && y > 0.445) {
					f->data[j*f->nx+i] = 0.33f;
				}
				
			}
		}
		break;
	case PARABOLIC_BASIN:
		std::cout << "Parabolic basin";
		f->dx = 8000.0f / (float) nx_;
		f->dy = 8000.0f / (float) ny_;

		for (unsigned int j = 0; j < f->ny; ++j) {
			float y = f->dy * j - 4000.0f;
			for (unsigned int i = 0; i < f->nx; ++i) {
				using namespace parabolic_basin;
				float x = f->dx * i - 4000.0f;
				f->data[j*f->nx+i] = D0*((x*x+y*y)/(L*L) - 1.0f);
			}
		}
		break;
	case MGPU_TEST:
		f->dx = 100.0f / (float) nx_;
		f->dy = 100.0f / (float) ny_;
		std::cout << "MGPU test case";
		for (unsigned int i=0; i<f->nx*f->ny; ++i)
			f->data[i] = 0.f;
		break;
	default:
		std::cout << "Could not recognize " << no << " as a valid id." << std::endl;
		exit(-1);
	case 10:
		std::cout << "Halvor-test";
		/*
		f->dx = 1;
		f->dy = 1;
		*/

		for (unsigned int j = 0; j < f->ny; ++j) {
			for (unsigned int i = 0; i < f->nx; ++i) {
				f->data[j*f->nx+i] = 100*i/((float) f->nx);
			}
		}
		break;
	}

	std::cout << "' (" << f->nx << "x" << f->ny << " values)" << std::endl;

	return f;
}

inline boost::shared_ptr<Field> generate_water_elevation(int no, int nx_, int ny_) {
	boost::shared_ptr<Field> f;

	if (nx_ <= 0 || ny_ <= 0) {
		std::cout << "Invalid nx or ny: [" << nx_ << ", " << ny_ << "]." << std::endl;
		exit(-1);
	}

	std::cout << "Generating water elevation: '";

	f.reset(new Field(nx_, ny_));

	switch (no) {
	case 0:
		std::cout << "column_dry";
#pragma omp parallel for
		for (int j=0; j<static_cast<int>(f->ny); ++j) {
			float y = (j+0.5f) / (float) f->ny - 0.5f;
			for (unsigned int i=0; i<f->nx; ++i) {
				float x = (i+0.5f) / (float) f->nx - 0.5f;
				if ((x*x)+(y*y)<0.01f)
					f->data[j*f->nx+i] = 1.0f;
				else
					f->data[j*f->nx+i] = 0.0f;
			}
		}
		break;


	case 1:
		std::cout << "gaussian";
#pragma omp parallel for
		for (int j=0; j<static_cast<int>(f->ny); ++j) {
			float y = (j+0.5f) / (float) f->ny - 0.5f;
			for (unsigned int i=0; i<f->nx; ++i) {
				float x = (i+0.5f) / (float) f->nx - 0.5f;
				//if ((x*x)+(y*y)<0.01)
					f->data[j*f->nx+i] = exp(-(x*x) - (y*y));
				//else
				//	f->data[j*f->nx+i] = 0.0f;
			}
		}
		break;



	case 2:
		std::cout << "zero";
#pragma omp parallel for
		for (int i=0; i<static_cast<int>(f->nx*f->ny); ++i)
			f->data[i] = 0.0f;
		break;


	case EU_CADAM:
		std::cout << "EU CADAM";
		f->dx = 38.0f / (float) f->nx;
		f->dy = 20.0f / (float) f->ny;
#pragma omp parallel for
		for (int j=0; j<static_cast<int>(f->ny); ++j) {
			for (unsigned int i = 0; i<f->nx; ++i) {
				if ((i+0.5)*f->dx <= 15.5f) {
					f->data[j*f->nx+i] = 0.75f;
				}
				else {
					f->data[j*f->nx+i] = 0.0f;
				}
			}
		}
		break;


	case STEADY_FLOW_OVER_BUMP:
		std::cout << "Steady Flow Over Bump";
		f->dx = 25.0f / (float) f->nx;
		f->dy = 20.0f / (float) f->ny;
#pragma omp parallel for
		for (int i = 0; i<static_cast<int>(f->nx*f->ny); ++i)
			f->data[i] = 1.0f;
		break;


	case 5:
		std::cout << "column_wet";
#pragma omp parallel for
		for (int j=0; j<static_cast<int>(f->ny); ++j) {
			float y = (j+0.5f) / (float) f->ny - 0.5f;
			for (unsigned int i=0; i<f->nx; ++i) {
				float x = (i+0.5f) / (float) f->nx - 0.5f;
				if ((x*x)+(y*y)<0.01f)
					f->data[j*f->nx+i] = 1.0f;
				else
					f->data[j*f->nx+i] = 0.1f;
			}
		}
		break;
		
	case IDEALISED_CIRCULAR_DAM:
		std::cout << "Idealised Circular Dam";
		f->dx = 100.0f / (float) f->nx;
		f->dy = 100.0f / (float) f->ny;
#pragma omp parallel for
		for (int j=0; j<static_cast<int>(f->ny); ++j) {
			float y = f->dy*(j+0.5f)-50.0f;
			for (unsigned int i=0; i<f->nx; ++i) {
				float x = f->dx*(i+0.5f)-50.0f;
				if (sqrt(x*x+y*y) < 6.5f)
					f->data[j*f->nx+i] = 10.0f;
				else
					f->data[j*f->nx+i] = 0.0f;
			}
		}
		break;
	case CADAM_TEST_1:
		f->dx = 9.575f / (float) nx_;
		f->dy = 3.73f / (float) ny_;
		std::cout << "CADAM Test 1";
		
#pragma omp parallel for
		for (int j = 0; j < static_cast<int>(f->ny); ++j) {
			float y = f->dy * (j+0.5f);
			for (unsigned int i = 0; i < f->nx; ++i) {
				float x = f->dx * (i+0.5f);
				
				if(x < 2.39f && y < 2.44f) {
					// set reservoir water elevation
					f->data[j*f->nx+i] = 0.58f;
				} else {
					// set canal water elevation
					f->data[j*f->nx+i] = 0.34f;
				}
			}
		}

		break;


	case PARABOLIC_BASIN:
		std::cout << "Parabolic basin";
		f->dx = 8000.0f / (float) nx_;
		f->dy = 8000.0f / (float) ny_;

#pragma omp parallel for
		for (int j = 0; j < static_cast<int>(f->ny); ++j) {
			float y = f->dy * (j+0.5f) - 4000.0f;
			for (unsigned int i = 0; i < f->nx; ++i) {
				using namespace parabolic_basin;
				float x = f->dx * (i+0.5f) - 4000.0f;
				f->data[j*f->nx+i] = 2.0f*A*D0/L*(cos(omega*t)*x/L + sin(omega*t)*y/L + B);
			}
		}
		break;
	case MGPU_TEST:
		std::cout << "MGPU test case";
		f->dx = 100.0f / (float) f->nx;
		f->dy = 100.0f / (float) f->ny;
	#pragma omp parallel for
		for (int j=0; j<static_cast<int>(f->ny); ++j) {
			for (unsigned int i=0; i<f->nx; ++i) {
				if(j<5) {
					f->data[j*f->nx+i] = 2.f;
				} else {
					f->data[j*f->nx+i] = 1.f;
				}
			}
		}
		break;
    case 10:
        std::cout << "Wet";
            for (int i=0; i<static_cast<int>(f->nx*f->ny); ++i)
                f->data[i] = 0.3f;
        break;

	default:
		std::cout << "Could not recognize " << no << " as a valid id." << std::endl;
		exit(-1);
	}

	std::cout << "' (" << f->nx << "x" << f->ny << " values)" << std::endl;

	return f;
}


inline boost::shared_ptr<Field> generate_u_discharge(int no, int nx_, int ny_) {
	boost::shared_ptr<Field> f;

	if (nx_ <= 0 || ny_ <= 0) {
		std::cout << "Invalid nx or ny: [" << nx_ << ", " << ny_ << "]." << std::endl;
		exit(-1);
	}

	std::cout << "Generating u discharge: '";

	f.reset(new Field(nx_, ny_));

	switch (no) {
	case 0:
		std::cout << "zero";
#pragma omp parallel for
		for (int j=0; j<static_cast<int>(f->ny); ++j)
			for (unsigned int i=0; i<f->nx; ++i)
				f->data[j*f->nx+i] = 0.0f;
		break;

	case PARABOLIC_BASIN:
		std::cout << "Parabolic basin";
		f->dx = 8000.0f / (float) nx_;
		f->dy = 8000.0f / (float) ny_;

#pragma omp parallel for
		for (int j = 0; j < static_cast<int>(f->ny); ++j) {
			float y = f->dy * (j+0.5f) - 4000.0f;
			for (unsigned int i = 0; i < f->nx; ++i) {
				using namespace parabolic_basin;
				float x = f->dx * (i+0.5f) - 4000.0f;
				float Bx = D0*((x*x+y*y)/(L*L) - 1.0f);
				float w = 2.0f*A*D0/L*(cos(omega*t)*x/L + sin(omega*t)*y/L + B);
				f->data[j*f->nx+i] = -(w-Bx)*A*omega*sin(omega*t);
			}
		}
		break;

	default:
		std::cout << "Could not recognize " << no << " as a valid id." << std::endl;
		exit(-1);
	}

	std::cout << "' (" << f->nx << "x" << f->ny << " values)" << std::endl;

	return f;
}


inline boost::shared_ptr<Field> generate_v_discharge(int no, int nx_, int ny_) {
	boost::shared_ptr<Field> f;

	if (nx_ <= 0 || ny_ <= 0) {
		std::cout << "Invalid nx or ny: [" << nx_ << ", " << ny_ << "]." << std::endl;
		exit(-1);
	}

	std::cout << "Generating v discharge: '";

	f.reset(new Field(nx_, ny_));

	switch (no) {
	case 0:
		std::cout << "zero";
#pragma omp parallel for
		for (int j=0; j<static_cast<int>(f->ny); ++j)
			for (unsigned int i=0; i<f->nx; ++i)
				f->data[j*f->nx+i] = 0.0f;
		break;

	case PARABOLIC_BASIN:
		std::cout << "Parabolic basin";
		f->dx = 8000.0f / (float) nx_;
		f->dy = 8000.0f / (float) ny_;

#pragma omp parallel for
		for (int j = 0; j <static_cast<int>(f->ny); ++j) {
			float y = f->dy * (j+0.5f) - 4000.0f;
			for (unsigned int i = 0; i < f->nx; ++i) {
				using namespace parabolic_basin;
				float x = f->dx * (i+0.5f) - 4000.0f;
				float Bx = D0*((x*x+y*y)/(L*L) - 1.0f);
				float w = 2.0f*A*D0/L*(cos(omega*t)*x/L + sin(omega*t)*y/L + B);
				f->data[j*f->nx+i] = (w-Bx)*A*omega*cos(omega*t);
			}
		}
		break;

	default:
		std::cout << "Could not recognize " << no << " as a valid id." << std::endl;
		exit(-1);
	}

	std::cout << "' (" << f->nx << "x" << f->ny << " values)" << std::endl;

	return f;
}

inline boost::shared_ptr<Field> generate_manning_coefficient(int no, int nx_, int ny_) {
	boost::shared_ptr<Field> f;

	if (nx_ <= 0 || ny_ <= 0) {
		std::cout << "Invalid nx or ny: [" << nx_ << ", " << ny_ << "]." << std::endl;
		exit(-1);
	}

	std::cout << "Generating manning coefficient: '";

	f.reset(new Field(nx_, ny_));

	switch (no) {
	case 0:
	default:
		std::cout << "Could not recognize " << no << " as a valid id." << std::endl;
		exit(-1);
	}

	std::cout << "' (" << f->nx << "x" << f->ny << " values)" << std::endl;

	return f;
}

#endif /* DATA_GENERATION_HPP_ */
