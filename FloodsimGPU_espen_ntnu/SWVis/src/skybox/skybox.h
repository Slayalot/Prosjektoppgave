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


#ifndef SKYBOX_H__
#define SKYBOX_H__

namespace north {
#include "skybox_north.h"
}
namespace south {
#include "skybox_south.h"
}
namespace east {
#include "skybox_east.h"
}
namespace west {
#include "skybox_west.h"
}
namespace up {
#include "skybox_up.h"
}
namespace down {
#include "skybox_down.h"
}

static const unsigned int skybox_width  = north::width;
static const unsigned int skybox_height = north::height;
static const unsigned char* skybox_north = north::pixel_data[0];
static const unsigned char* skybox_south = south::pixel_data[0];
static const unsigned char* skybox_east  = east::pixel_data[0];
static const unsigned char* skybox_west  = west::pixel_data[0];
static const unsigned char* skybox_up    = up::pixel_data[0];
static const unsigned char* skybox_down  = down::pixel_data[0];



#endif
