#############################################################################
#                                                                           #
#                                                                           #
# (c) Copyright 2010, 2011, 2012 by                                         #
#     SINTEF, Oslo, Norway                                                  #
#     All rights reserved.                                                  #
#                                                                           #
#  THIS SOFTWARE IS FURNISHED UNDER A LICENSE AND MAY BE USED AND COPIED    #
#  ONLY IN  ACCORDANCE WITH  THE  TERMS  OF  SUCH  LICENSE  AND WITH THE    #
#  INCLUSION OF THE ABOVE COPYRIGHT NOTICE. THIS SOFTWARE OR  ANY  OTHER    #
#  COPIES THEREOF MAY NOT BE PROVIDED OR OTHERWISE MADE AVAILABLE TO ANY    #
#  OTHER PERSON.  NO TITLE TO AND OWNERSHIP OF  THE  SOFTWARE IS  HEREBY    #
#  TRANSFERRED.                                                             #
#                                                                           #
#  SINTEF  MAKES NO WARRANTY  OF  ANY KIND WITH REGARD TO THIS SOFTWARE,    #
#  INCLUDING,   BUT   NOT   LIMITED   TO,  THE  IMPLIED   WARRANTIES  OF    #
#  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.                    #
#                                                                           #
#  SINTEF SHALL NOT BE  LIABLE  FOR  ERRORS  CONTAINED HEREIN OR DIRECT,    #
#  SPECIAL,  INCIDENTAL  OR  CONSEQUENTIAL  DAMAGES  IN  CONNECTION WITH    #
#  FURNISHING, PERFORMANCE, OR USE OF THIS MATERIAL.                        #
#                                                                           #
#                                                                           #
#############################################################################

# - Project stuff
# Including name, placement of binaries, where to find the modules we
# include, the modules we include, global vars (include dirs, lib dirs, etc)
PROJECT(KPSIMULATORAPPS)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)

#Set where executables end up
IF(NOT EXECUTABLE_OUTPUT_PATH)
	SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
ENDIF(NOT EXECUTABLE_OUTPUT_PATH)

FIND_PACKAGE(Boost REQUIRED COMPONENTS filesystem system thread program_options)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(OpenMP REQUIRED)
FIND_PACKAGE(OpenGL REQUIRED)
FIND_PACKAGE(GLUT REQUIRED)
FIND_PACKAGE(GLEW REQUIRED)
FIND_PACKAGE(NetCDF QUIET)

IF(GLUT_INCLUDE_DIR MATCHES "NOTFOUND")
	MESSAGE(FATAL_ERROR "GLUT not found")
ENDIF()

OPTION(KPSIMULATORAPPS_USE_NETCDF "Enable or disable NetCDF-support" TRUE)
OPTION(KPSIMULATORAPPS_USE_NETCDF_COMPRESSION "Enable or disable NetCDF compression-support (Requires new version of netcdf)" TRUE)
OPTION(KPSIMULATORAPPS_USE_VISUALIZER_WITH_BOUNDARIES "Show boundaries when visualizing?" FALSE)

# - Find the KPSimulator include dir
FIND_PATH(KPSIMULATOR_INCLUDE_DIR NAMES KPSimulator.h DOC "Path to directory where KPSimulator.h can be found")
FIND_LIBRARY(KPSIMULATOR_LIBRARIES NAMES KPSimulator.lib KPSimulator.a DOC "Path to directory where KPSimulator libraries can be found")

# - Find visualizer
FIND_PATH(SWVIS_INCLUDE_DIR NAMES SWVisualizer.h DOC "Path to directory where SWVisualizationContext.h can be found")
FIND_LIBRARY(SWVIS_LIBRARIES NAMES SWVisualizer.lib SWVisualizer.a DOC "Path to directory where visualizer library can be found")

INCLUDE_DIRECTORIES(
	${KPSIMULATORAPPS_SOURCE_DIR}/include
	${KPSIMULATOR_INCLUDE_DIR}
	${SWVIS_INCLUDE_DIR}
	${GLUT_INCLUDE_DIR} #Must be included before CUDA_SDK, as the SDK includes Kilgards version...
	${GLEW_INCLUDE_DIR}
	${NetCDF_INCLUDE_DIR}
	${CUDA_TOOLKIT_ROOT_DIR}/include
	${Boost_INCLUDE_DIR}
	${CMAKE_CURRENT_BINARY_DIR})
	
IF(KPSIMULATORAPPS_USE_NETCDF)
	INCLUDE_DIRECTORIES(${NetCDF_INCLUDE_DIR})
ENDIF()

CUDA_INCLUDE_DIRECTORIES(
	${KPSIMULATORAPPS_SOURCE_DIR}/include
	${KPSIMULATOR_INCLUDE_DIR}
	${SWVIS_INCLUDE_DIR}
	${GLUT_INCLUDE_DIR} #Must be included before CUDA_SDK, as the SDK includes Kilgards version...
	${GLEW_INCLUDE_DIR}
	${NetCDF_INCLUDE_DIR}
	${CUDA_TOOLKIT_ROOT_DIR}/include
	${Boost_INCLUDE_DIR}
	${CMAKE_CURRENT_BINARY_DIR})

LINK_DIRECTORIES(
	${SWVIS_LIB_DIR} 
	${Boost_LIBRARY_DIRS})

#Create configure.h from configure.h.in
#file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/configure.h")
configure_file("include/configure.h.in" "${CMAKE_CURRENT_BINARY_DIR}/configure.h" ESCAPE_QUOTES)

# - Create objects needed
FILE(GLOB KPSIMULATORAPPS_src "src/*.cpp")
ADD_LIBRARY(KPSimulatorApps STATIC ${KPSIMULATORAPPS_src})
IF(NOT CMAKE_HOST_WIN32)
	SET_TARGET_PROPERTIES(KPSimulatorApps PROPERTIES COMPILE_FLAGS "${OpenMP_CXX_FLAGS}")
ENDIF()

# - Create applications
SET(VISUALIZER_TARGETS app_kp_visualization app_nc_visualization)
FILE(GLOB KPSIMULATORAPPS_app src/app/*.cpp)
FOREACH(filename ${KPSIMULATORAPPS_app})
  GET_FILENAME_COMPONENT(target ${filename} NAME_WE)
  SET(name "app_${target}")
  ADD_EXECUTABLE(${name} ${filename})
  TARGET_LINK_LIBRARIES(${name} 
	KPSimulatorApps
	${KPSIMULATOR_LIBRARIES}
	${CUDA_LIBRARIES})
	
IF(KPSIMULATORAPPS_USE_NETCDF)
  TARGET_LINK_LIBRARIES(${name} ${NetCDF_LIBRARIES})
ENDIF()
  IF(VISUALIZER_TARGETS MATCHES "${target}")
	TARGET_LINK_LIBRARIES(${name}
  SWVis
  ${OPENGL_LIBRARIES} 
  ${GLUT_LIBRARIES}
  ${GLEW_LIBRARY}
)
  ENDIF()
  IF(NOT CMAKE_HOST_WIN32)
    TARGET_LINK_LIBRARIES(${name}	${Boost_LIBRARIES})
    SET_TARGET_PROPERTIES(${name} PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}")
  ENDIF()
  SET_TARGET_PROPERTIES(${name} PROPERTIES OUTPUT_NAME "${target}")
ENDFOREACH(filename ${KPSIMULATORAPPS_app})

