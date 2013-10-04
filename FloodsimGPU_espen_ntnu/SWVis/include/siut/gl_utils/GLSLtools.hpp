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

#ifndef _SIUT_GL_UTILS_GLSLTOOLS_HPP
#define _SIUT_GL_UTILS_GLSLTOOLS_HPP

#ifdef USE_GLEW
#include <GL/glew.h>
#else
#ifndef GL_GLEXT_PROTOTYPES
#define GL_GLEXT_PROTOTYPES
#endif
#include <GL/gl.h>
#include <GL/glu.h>
#endif

//#define DEBUG_GL

#if defined (CHECK_GL) || defined (DEBUG_GL)
#define GL_TESTING
#include <sstream>
#include <iostream>
#include "siut/io_utils/snarf.hpp"
#ifdef DEBUG_GL
#include <stdexcept>
#endif
#endif


#include <iostream>
#include <vector>
#include <string>
#include "siut/io_utils/snarf.hpp"

#define FBO_ERROR(a) case a: printThrowFBOError(where, #a); break
namespace siut
{
  /** \brief Namespace with tools for compiling GLSL shaders, check FBOs for completeness,
   * printing GL-errors and classes that are thrown if DEBUG_GL is defined.
   * If CHECK_GL is defined, will dump messages to stderr, whereas if DEBUG_GL is defined, will throw them with the relevant error class.
   */
  namespace gl_utils
  {
   


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///// Error-classes                                                                                                       /////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /** Shader error class. 
     * Thrown if an error is encountered when compiling a shader. 
     *Contains the infolog about what went wrong, as well as the source code that threw the error.
     */
    class shader_error : public std::runtime_error
    {      
    public:
      explicit shader_error(const std::string &what)
	: std::runtime_error(what)
      {
	;
      }
      
 
      virtual ~shader_error() throw()
      {
	;
      }

    };


    /** GL error class, thrown when printGLError() encounters an error and DEBUG_GL is defined. */
    class gl_error : public std::runtime_error
    {
    public:
      explicit gl_error(const std::string &what)
	: std::runtime_error(what)
      {
	;
      }

      virtual ~gl_error() throw()
      {
	;
      }
    };

    /** FBO error class, thrown when checkFramebufferStatus() encounteres an error and DEBUG_GL is defined. */
    class fbo_error : public std::runtime_error
    {
    public:
      explicit fbo_error(const std::string &what)
	:runtime_error(what)
      {
	;
      }

      virtual ~fbo_error() throw()
      {
	;
      }

    };


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///// Functions                                                                                                           /////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** Prints or throw gl_error class if an error is encountered.
 * If CHECK_GL is defined, will dump errors to stderr.
 * If DEBUG_GL is defined, will throw gl_error with what error was encountered.
 * \param fname the filename of where the function is called from. (__FILE__)
 * \param line the line number of where it was called from. (__LINE__)
 *
 * \example printGLError(__FILE__, __LINE__);
 */
    inline void printGLError(std::string fname, int line)
    {      
        GLenum error = glGetError();
        if( error != GL_NO_ERROR ) {
            std::stringstream s;
	    s << fname << '@' << line << ": OpenGL error: "; 

            do {
                s <<  gluErrorString(error) << ". ";
                error = glGetError();
            } while( error != GL_NO_ERROR );

            throw gl_error( s.str() );
        }
    }

#define CHECK_GL do { siut::gl_utils::printGLError( __FILE__, __LINE__ ); } while(0)

    inline
    void linkProgram( GLuint program )
    {
      glLinkProgram( program );

      GLint linkstatus;
      glGetProgramiv( program, GL_LINK_STATUS, &linkstatus );
      if( linkstatus != GL_TRUE ) {

	std::string log;
	
        GLint logsize;
	glGetProgramiv( program, GL_INFO_LOG_LENGTH, &logsize );
        if( logsize > 0 ) {
            std::vector<GLchar> infolog( logsize+1 );
            glGetProgramInfoLog( program, logsize, NULL, &infolog[0] );
            log = std::string( infolog.begin(), infolog.end() );
        }
        else {
	  log = "Empty log.";
        }
	throw shader_error( "GLSL link failed:\n" + log );
      }
    }


    /** Compiles the shader of type from src, returns the shader id generated by OpenGL.
     * Compiles the shader from src to the type specified from type.
     * Uses printGLError to chek for errors creating the shader.
     * If DEBUG_GL is defined, will throw shader_error, containing error and source, if there is a problem compiling the shader.
     * If CHECK_GL is defined it will dump the error and source to stderr.
     * 
     * \param src source of the shader.
     * \param type type of shader to compile GL_VERTEX_SHADER, GL_FRAGMENT_SHADER etc.
     * \param return GLuint shader_id.
     */
    inline GLuint compileShader(const std::string &src, GLenum type, bool fail_on_warnings = false )
    {
      GLuint shader_id = glCreateShader(type);
      printGLError(__FILE__, __LINE__);

      const char *p = src.c_str();
      glShaderSource(shader_id, 1, &p, NULL);
      printGLError(__FILE__, __LINE__);
      glCompileShader(shader_id);
      printGLError(__FILE__, __LINE__);


      GLint llength;
      GLint status;

      glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, &llength);
      glGetShaderiv(shader_id, GL_COMPILE_STATUS, &status);
      if( (status != GL_TRUE) || (fail_on_warnings && (llength > 1) ) ) {
	
	std::string infol;
	if(llength > 1) {
	  GLchar* infolog = new GLchar[llength];
	  glGetShaderInfoLog(shader_id, llength, NULL, infolog);
	  infol = std::string( infolog );
	  delete[] infolog;
	}
	else {
	  infol = "no log.";
	}

	std::stringstream s;
	if( status != GL_TRUE ) {
	  s << "Compilation returned errors." << std::endl;
	}
	s << infol << std::endl;
	io_utils::dumpSourceWithLineNumbers( src, s );
	throw shader_error( s.str() );
	return 0;
      }	      
      return shader_id;
    }

    /** Method used by checkFramebufferStatus() to either throw or print if there is an error with the framebuffer.
     * Not really meant to be used elsewhere, but hey...
     * If DEBUG_GL is defined, will throw fbo_error with what has gone wrong.
     * If CHECK_GL is defined, will dump the error message to stderr.
     * \param where where checkFramebufferStatus() was called from
     * \param what what has gone wrong.
     */
    inline void printThrowFBOError(const std::string &where, const std::string &what)
    {
#if defined (DEBUG_GL) || defined(CHECK_GL)
      std::string ret("FBO Error in" + where + "\n" + "Problem is:\t" + what);
#ifdef DEBUG_GL
      throw new fbo_error(ret);
#else
      std::cerr << ret << std::endl;
#endif
#endif
    }

    /** Method used to check completeness of a framebuffer, fbo.
     * Checks if there is any problems with the completeness of the framebuffer, 
     * If a problem is encountered it calls printThrowFBOError() with the error that was encountered, and where it was called from.
     *
     * \param where where checkFramebufferStatus was called from. (__FILE__ + __LINE__)
     *
     */
    inline void checkFramebufferStatus(const std::string &where)
    {
#ifdef GL_ARB_framebuffer_object
		GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		if( status != GL_FRAMEBUFFER_COMPLETE ) {
			switch(status)
			{
				//	  FBO_ERROR(GL_FRAMEBUFFER_COMPLETE);
				FBO_ERROR(GL_FRAMEBUFFER_UNDEFINED);
				FBO_ERROR(GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT);
				FBO_ERROR(GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT);
				FBO_ERROR(GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER);
				FBO_ERROR(GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER);
				FBO_ERROR(GL_FRAMEBUFFER_UNSUPPORTED);
				FBO_ERROR(GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE);
			default:
				{
					std::stringstream s;
					s << "unknown fbo error " << status << ".\n";
					printThrowFBOError(where, s.str() );
				}
			}
		}
#else //NVidia driver for XP only exposes EXT-versions 2009-09-18
		GLenum status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
		if( status != GL_FRAMEBUFFER_COMPLETE_EXT ) {
			switch(status)
			{
				//	  FBO_ERROR(GL_FRAMEBUFFER_COMPLETE);
				//FBO_ERROR(GL_FRAMEBUFFER_UNDEFINED);
				FBO_ERROR(GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT);
				FBO_ERROR(GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT);
				FBO_ERROR(GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT);
				FBO_ERROR(GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT);
				FBO_ERROR(GL_FRAMEBUFFER_UNSUPPORTED_EXT);
				FBO_ERROR(GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE_EXT);
			default:
				{
					std::stringstream s;
					s << "unknown fbo error " << status << ".\n";
					printThrowFBOError(where, s.str() );
				}
			}
		}
#endif
    }

    inline
    GLint
    _getVaryingLocation( GLuint program, const char* name, const std::string& file, int line )
    {
      GLint loc = glGetVaryingLocationNV( program, name );
      if( loc == -1 ) {
	std::stringstream out;
	out << file << '@' << line<< ": failed to get location of varying \"" << name << "\".";
	throw std::runtime_error( out.str() );
      }
      return loc;
    }

#define getVaryingLocation(program,name) (siut::gl_utils::_getVaryingLocation((program),(name),__FILE__, __LINE__))


  }//end namespace GL
}
#endif
