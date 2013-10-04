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

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)
PROJECT(SWVIS)

#Set where executables end up
IF(NOT LIBRARY_OUTPUT_PATH)
	SET(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib)
ENDIF(NOT LIBRARY_OUTPUT_PATH)

#Extends the CMAKE_MODULE_PATH with paths where the user might place
#his/her own cmake modules on a "per user" basis
OPTION(EXTEND_CMAKE_MODULE_PATH "Extend the CMAKE_MODULE_PATH variable with user directories?" ON)
IF(EXTEND_CMAKE_MODULE_PATH)
	SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
		"${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules"
		"./cmake/Modules"
		"~/cmake/Modules")
ENDIF(EXTEND_CMAKE_MODULE_PATH)

FIND_PACKAGE(Boost REQUIRED)
FIND_PACKAGE(OpenGL REQUIRED)
FIND_PACKAGE(GLEW REQUIRED)
FIND_PACKAGE(GLUT REQUIRED)
FIND_PACKAGE(CUDA REQUIRED)

INCLUDE_DIRECTORIES(
	${CMAKE_CURRENT_SOURCE_DIR}/include
	${CMAKE_CURRENT_SOURCE_DIR}/src
	${Boost_INCLUDE_DIR}
	${GLUT_INCLUDE_DIR} #Must be included before CUDA_SDK, as the SDK includes Kilgards version...
	${GLEW_INCLUDE_DIR}
	${CUDA_TOOLKIT_ROOT_DIR}/include
  )
CUDA_INCLUDE_DIRECTORIES(
	${CMAKE_CURRENT_SOURCE_DIR}/include
	${CMAKE_CURRENT_SOURCE_DIR}/src
	${Boost_INCLUDE_DIR}
	${GLUT_INCLUDE_DIR} #Must be included before CUDA_SDK, as the SDK includes Kilgards version...
	${GLEW_INCLUDE_DIR}
	${CUDA_TOOLKIT_ROOT_DIR}/include
  )

#include_external_msproject(visualizer_src "${visualizer_SOURCE_DIR}/visualizer_src.vcproj")

#Add app that generates header files from glsl-files
ADD_EXECUTABLE(app_glslToH "src/app/glslToH.cpp")
SET_TARGET_PROPERTIES(app_glslToH PROPERTIES OUTPUT_NAME "glslToH")

ADD_EXECUTABLE(app_ppmToH "src/app/ppmToH.cpp")
SET_TARGET_PROPERTIES(app_ppmToH PROPERTIES OUTPUT_NAME "ppmToH")


SET(glsl_srcs "") 
FILE(GLOB SWVIS_vglsl data/glsl/*.glsl)
FOREACH(filename ${SWVIS_vglsl})
    GET_FILENAME_COMPONENT(target ${filename} NAME_WE)
    SET(name "${CMAKE_CURRENT_SOURCE_DIR}/src/glsl/${target}.h")
    SET(glsl_srcs ${glsl_srcs} ${name})
	GET_TARGET_PROPERTY(GLSL_TO_H_EXE app_glslToH LOCATION)
	ADD_CUSTOM_COMMAND(
	  OUTPUT ${name}
		COMMAND ${GLSL_TO_H_EXE} ${filename} ${name}
	  DEPENDS app_glslToH ${filename}
	)
ENDFOREACH(filename ${SWVIS_vglsl})

SET(env_srcs)
FILE(GLOB SWVIS_env data/env/*.ppm)
FOREACH(filename ${SWVIS_env})
  GET_FILENAME_COMPONENT(target ${filename} NAME_WE)
  SET(name "${CMAKE_CURRENT_SOURCE_DIR}/src/skybox/${target}.h")
  SET(env_srcs ${env_srcs} ${name})
	GET_TARGET_PROPERTY(PPM_TO_H_EXE app_ppmToH LOCATION)
	ADD_CUSTOM_COMMAND(
	  OUTPUT ${name}
		COMMAND ${PPM_TO_H_EXE} ${filename} ${name}
	  DEPENDS app_ppmToH ${filename}
	)
ENDFOREACH()

FILE(GLOB srcs src/*.cpp)
ADD_LIBRARY(SWVis ${srcs} ${glsl_srcs} ${env_srcs})




