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


#ifndef SHADERS_H__
#define SHADERS_H__

namespace landscape_fragment {
#include "glsl/Landscape_fragment.h"
}
namespace landscape_vertex {
#include "glsl/Landscape_vertex.h"
}
namespace water_fresnel_fragment {
#include "glsl/Water_fresnel_fragment.h"
}
namespace water_fresnel_vertex {
#include "glsl/Water_fresnel_vertex.h"
}
namespace water_depth_fragment {
#include "glsl/Water_depth_fragment.h"
}
namespace water_depth_vertex {
#include "glsl/Water_depth_vertex.h"
}
namespace water_velocity_fragment {
#include "glsl/Water_velocity_fragment.h"
}
namespace water_velocity_vertex {
#include "glsl/Water_velocity_vertex.h"
}

#endif
