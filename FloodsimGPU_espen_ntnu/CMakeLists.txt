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


cmake_minimum_required(VERSION 2.8)
PROJECT(FloodSimGPU)

SET(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/Modules" CACHE STRING " ")

IF(NOT CMAKE_BUILD_TYPE)
	SET(CMAKE_BUILD_TYPE "Release" CACHE STRING " " FORCE)
ENDIF()
IF(NOT LIBRARY_OUTPUT_PATH)
  SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
ENDIF()
IF(NOT EXECUTABLE_OUTPUT_PATH)
  SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
ENDIF()

#Set compilation flags
IF(MSVC10)
	#Enable multiprocessor compilation for speed
	IF(NOT CMAKE_CXX_FLAGS MATCHES "/MP")
		SET(CMAKE_CXX_FLAGS " /MP ${CMAKE_CXX_FLAGS}" CACHE STRING " " FORCE)
	ENDIF()
ENDIF()
IF(NOT CMAKE_HOST_WIN32)
	# Enable position independent code for linux build
	IF(NOT CMAKE_CXX_FLAGS MATCHES "-fPIC")
		SET(CMAKE_CXX_FLAGS " -fPIC ${CMAKE_CXX_FLAGS}" CACHE STRING " " FORCE)
	ENDIF()
ENDIF()


SET(Boost_USE_MULTITHREADED ON)

#Set the paths we know
FIND_LIBRARY(KPSIMULATOR_LIBRARY_DEBUG
	NAMES "KPSimulatord.lib" "libKPSimulatord.a"
	PATHS "${PROJECT_SOURCE_DIR}/KPSimulator/lib/")
FIND_LIBRARY(KPSIMULATOR_LIBRARY_RELEASE
	NAMES "KPSimulator.lib" "libKPSimulator.a"
	PATHS "${PROJECT_SOURCE_DIR}/KPSimulator/lib/")	
IF(NOT KPSIMULATOR_LIBRARY_DEBUG AND NOT KPSIMULATOR_LIBRARY_RELEASE) 
	SET(KPSIMULATOR_LIBRARIES KPSimulator CACHE FILEPATH " " FORCE)
	LINK_DIRECTORIES(${LIBRARY_OUTPUT_PATH})
ELSE()
	SET(KPSIMULATOR_LIBRARIES debug ${KPSIMULATOR_LIBRARY_DEBUG} optimized ${KPSIMULATOR_LIBRARY_RELEASE} CACHE FILEPATH "Path to KPSimulator libraries")
ENDIF()

SET(KPSIMULATOR_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/KPSimulator/include" CACHE PATH " ")
SET(SWVIS_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/SWVis/include" CACHE PATH " ")
SET(SWVIS_LIB_DIR "${LIBRARY_OUTPUT_PATH}" CACHE PATH " ")

ADD_SUBDIRECTORY(KPSimulator)
ADD_SUBDIRECTORY(SWVis)
ADD_SUBDIRECTORY(KPSimulatorApps)

INSTALL(DIRECTORY . DESTINATION .
	PATTERN ".svn" EXCLUDE
	PATTERN "build*" EXCLUDE
	PATTERN "KPSimulator" EXCLUDE)
INSTALL(CODE "FILE(WRITE \"${CMAKE_INSTALL_PREFIX}/KPSimulator/CMakelists.txt\" \" \")")
