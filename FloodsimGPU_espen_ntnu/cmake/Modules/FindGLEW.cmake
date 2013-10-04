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


#Find glew library
FIND_LIBRARY(GLEW_LIBRARY 
  NAMES GLEW glew glew32
  PATHS "/usr/lib"
  "/usr/lib64"
  "$ENV{ProgramFiles}/Microsoft Visual Studio 8/VC/PlatformSDK/Lib"
  "$ENV{ProgramFiles}/Microsoft Visual Studio 9.0/VC/lib/"
  "$ENV{PROGRAMW6432}/Microsoft SDKs/Windows/v6.0A/Lib"
  )

#Find glew header
FIND_PATH(GLEW_INCLUDE_DIR "GL/glew.h"
  "/usr/include"
  "$ENV{ProgramFiles}/Microsoft Visual Studio 8/VC/PlatformSDK/Include"
  "$ENV{ProgramFiles}/Microsoft Visual Studio 9.0/VC/include/"
  "$ENV{PROGRAMW6432}/Microsoft SDKs/Windows/v6.0A/Include"
)

#check that we have found everything
SET(GLEW_FOUND "NO")
IF(GLEW_LIBRARY)
  IF(GLEW_INCLUDE_DIR)
    SET(GLEW_FOUND "YES")
  ENDIF(GLEW_INCLUDE_DIR)
ENDIF(GLEW_LIBRARY)

#Mark options as advanced
MARK_AS_ADVANCED(
  GLEW_INCLUDE_DIR
  GLEW_LIBRARY
)
