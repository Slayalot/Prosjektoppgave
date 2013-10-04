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
PROJECT(KPSIMULATOR)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

# - Find packages we need
FIND_PACKAGE(Boost REQUIRED COMPONENTS unit_test_framework)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(OpenMP REQUIRED)
FIND_PACKAGE(Mercurial)

IF(NOT LIBRARY_OUTPUT_PATH)
	SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
ENDIF(NOT LIBRARY_OUTPUT_PATH)
	
INCLUDE_DIRECTORIES(
	${KPSIMULATOR_SOURCE_DIR}/src/include
	${KPSIMULATOR_SOURCE_DIR}/include
	${Boost_INCLUDE_DIR}
	${CUDA_TOOLKIT_ROOT_DIR}/include
	${CMAKE_CURRENT_BINARY_DIR})
CUDA_INCLUDE_DIRECTORIES(
	${KPSIMULATOR_SOURCE_DIR}/src/include
	${KPSIMULATOR_SOURCE_DIR}/include
	${Boost_INCLUDE_DIR}
	${CUDA_TOOLKIT_ROOT_DIR}/include
	${CMAKE_CURRENT_BINARY_DIR}C)

LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})

IF(NOT CUDA_NVCC_FLAGS)
    SET(CUDA_NVCC_FLAGS "-ftz=true -prec-div=false -prec-sqrt=false" CACHE STRING " " FORCE)
ENDIF(NOT CUDA_NVCC_FLAGS)

#Get the version number from subversion
IF (MERCURIAL_FOUND) 
	MERCURIAL_HG_INFO(${PROJECT_SOURCE_DIR} KP_MERCURIAL)
	SET(REVISION ${KP_MERCURIAL_HG_REVNUMBER})# FIXME: Should also include${KP_MERCURIAL_HG_CHANGESET} or ${KP_MERCURIAL_HG_ID}
	STRING(REGEX MATCH "^[0-9]+" REVISION "${REVISION}")
ENDIF()


SET(KPSIMULATOR_MAJOR_VERSION "1" CACHE STRING "Specify the major version" FORCE)
SET(KPSIMULATOR_MINOR_VERSION "50" CACHE STRING "Specify the minor version" FORCE)
SET(KPSIMULATOR_BUILD_VERSION "${REVISION}" CACHE STRING "Specify the build version" FORCE)
SET(KPSIMULATOR_ZERO_FLUX_EPS 1.0e-7f CACHE STRING "All epsilon values less than this are interpreted as dry for the flux calculation" FORCE)
SET(KPSIMULATOR_DRY_EPS 1.0e-7f CACHE STRING "All epsilon values less than this are interpreted as dry" FORCE)
SET(KPSIMULATOR_FLUX_BLOCK_WIDTH 16 CACHE STRING "Width of CUDA-blocks in flux and RK calculations")
SET(KPSIMULATOR_FLUX_BLOCK_HEIGHT 12 CACHE STRING "Height of CUDA-blocks in flux and RK calculations")
SET(KPSIMULATOR_GROW_BLOCK_WIDTH 16 CACHE STRING "Width of CUDA-blocks in grow patches-kernel")
SET(KPSIMULATOR_GROW_BLOCK_HEIGHT 16 CACHE STRING "Height of CUDA-blocks in grow patches-kernel")
SET(KPSIMULATOR_FLUX_SLOPE_EPS 1.0e-3f CACHE STRING "Slopes less than this are considered flat in flux kernel" FORCE)
SET(KPSIMULATOR_MINMOD_THETA 1.3f CACHE STRING "Theta parameter for minmod slope limiter in flux kernel")

OPTION(KPSIMULATOR_USE_BC_SMALL_DOMAINS "Enable efficient execution for small domains? Do not enable unless you know the consequences" FALSE)
OPTION(KPSIMULATOR_USE_BC_FIXED_DEPTH_N "Enable or disable fixed depth boundary conditions" TRUE)
OPTION(KPSIMULATOR_USE_BC_FIXED_DEPTH_S "Enable or disable fixed depth boundary conditions" TRUE)
OPTION(KPSIMULATOR_USE_BC_FIXED_DEPTH_E "Enable or disable fixed depth boundary conditions" TRUE)
OPTION(KPSIMULATOR_USE_BC_FIXED_DEPTH_W "Enable or disable fixed depth boundary conditions" TRUE)
OPTION(KPSIMULATOR_USE_BC_FIXED_DISCHARGE_N "Enable or disable fixed discharge boundary conditions" TRUE)
OPTION(KPSIMULATOR_USE_BC_FIXED_DISCHARGE_S "Enable or disable fixed discharge boundary conditions" TRUE)
OPTION(KPSIMULATOR_USE_BC_FIXED_DISCHARGE_E "Enable or disable fixed discharge boundary conditions" TRUE)
OPTION(KPSIMULATOR_USE_BC_FIXED_DISCHARGE_W "Enable or disable fixed discharge boundary conditions" TRUE)
OPTION(KPSIMULATOR_USE_BC_OPEN_N "Enable or disable open boundary conditions" TRUE)
OPTION(KPSIMULATOR_USE_BC_OPEN_S "Enable or disable open boundary conditions" TRUE)
OPTION(KPSIMULATOR_USE_BC_OPEN_E "Enable or disable open boundary conditions" TRUE)
OPTION(KPSIMULATOR_USE_BC_OPEN_W "Enable or disable open boundary conditions" TRUE)
OPTION(KPSIMULATOR_USE_BC_WALL_N "Enable or disable wall boundary conditions" TRUE)
OPTION(KPSIMULATOR_USE_BC_WALL_S "Enable or disable wall boundary conditions" TRUE)
OPTION(KPSIMULATOR_USE_BC_WALL_E "Enable or disable wall boundary conditions" TRUE)
OPTION(KPSIMULATOR_USE_BC_WALL_W "Enable or disable wall boundary conditions" TRUE)
OPTION(KPSIMULATOR_USE_AUTOTUNE "Enable early exit autotuning optimization" TRUE)
OPTION(KPSIMULATOR_USE_FLUX_ROW "Use the row implementation of the flux kernel?" FALSE)

IF(${Subversion_FOUND})
    Subversion_WC_INFO(${KPSIMULATOR_SOURCE_DIR} KPSIMULATOR)
    SET(KPSIMULATOR_SIMULATOR_VERSION "${KPSIMULATOR_WC_REVISION}" CACHE STRING "Specify what version of KPSimulator this is" FORCE)
ENDIF()

IF(CMAKE_SIZEOF_VOID_P EQUAL 8)
	SET(KPSIMULATOR_TARGET_CPU_ARCH "64")
ELSEIF(CMAKE_SIZEOF_VOID_P EQUAL 4)
	SET(KPSIMULATOR_TARGET_CPU_ARCH "32")
ELSE()
	MESSAGE(ERROR "Could not deduce target architecture: sizeof(void*) is neither 4 or 8...")
ENDIF()
IF(CUDA_64_BIT_DEVICE_CODE)
	SET(KPSIMULATOR_TARGET_GPU_ARCH "64")
ELSE()
	SET(KPSIMULATOR_TARGET_GPU_ARCH "32")
ENDIF()
string(REGEX REPLACE "[.]" "" KPSIMULATOR_CUDA_VERSION ${CUDA_VERSION})

IF(KPSIMULATOR_SIMULATOR_VERSION STREQUAL "")
    MESSAGE(FATAL_ERROR "KPSIMULATOR_SIMULATOR_VERSION not defined")
ENDIF()

#Create configure.h from configure.h.in
#file(REMOVE "${KPSIMULATOR_SOURCE_DIR}/include/configure.h")
configure_file("src/include/configure.h.in" "${CMAKE_CURRENT_BINARY_DIR}/configure.h" ESCAPE_QUOTES)

# - Create the KPSimulator lib
# Adding the cuda files uses the cuda_compile macro, which has a bug on windows for cmake 2.8.3 and 2.8.6 at least
# in which GPUVideo_cu_compiled_src returns an empty list. To bypass this, simply set 
# CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE to false. Mail sent to James Biegler, NVIDIA about it.
SET(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE false)
CUDA_COMPILE(KPSimulatorBC_cu_compiled_src src/KPSimulatorBC.cu)
CUDA_COMPILE(KPSimulator_cu_compiled_src src/KPSimulator.cu)
ADD_LIBRARY(KPSimulator STATIC ${KPSimulator_cu_compiled_src} src/KPInitialConditions.cpp ${KPSimulatorBC_cu_compiled_src})
SET_TARGET_PROPERTIES(KPSimulator PROPERTIES DEBUG_POSTFIX "d")
SET_TARGET_PROPERTIES(KPSimulator PROPERTIES COMPILE_FLAGS "${OpenMP_CXX_FLAGS}")
INSTALL(TARGETS KPSimulator DESTINATION KPSimulator/lib)
INSTALL(DIRECTORY include DESTINATION KPSimulator PATTERN ".svn" EXCLUDE)


