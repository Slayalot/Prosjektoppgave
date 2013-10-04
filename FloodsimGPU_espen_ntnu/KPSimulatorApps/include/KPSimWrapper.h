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

#ifndef KPSIMWRAPPER_H__
#define KPSIMWRAPPER_H__

#include "SWVisualizationContext.h"
#include "gpu_ptr.hpp"

class KPSimWrapper : public SimulatorWrapper {
public:
	KPSimWrapper(KPInitialConditions& ic) {
		sim.reset(new KPSimulator(ic));
		std::cout << sim << std::endl;
	}
	virtual void step() {sim->step();}
	virtual float getTime() {return sim->getTime();}
	virtual float getDt() {return sim->getDt();}
	virtual int getTimeStep() {return sim->getTimeSteps();}
	virtual boost::shared_ptr<gpu_ptr_2D<float> > getU1() {return sim->getU1();}
	virtual boost::shared_ptr<gpu_ptr_2D<float> > getU2() {return sim->getU2();}
	virtual boost::shared_ptr<gpu_ptr_2D<float> > getU3() {return sim->getU3();}
	virtual boost::shared_ptr<gpu_ptr_2D<float> > getB() {return sim->getBm();}

	inline void drawDebugTexture() {
		const float scale = 0.3f;
		boost::shared_ptr<gpu_ptr_2D<float> > D = sim->getD();
		GLuint tex = GLFuncs::newGrayTexture(D->getWidth(), D->getHeight());
		SWVisualizationContext::memcpyCudaToOgl(tex, 0, 0, 
			D->getRawPtr().ptr, D->getRawPtr().pitch, 
			D->getWidth()*sizeof(float), D->getHeight());

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, tex);
		glEnable(GL_TEXTURE_2D);

		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glLoadIdentity();
		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
		glOrtho(-0.5, 0.5, -0.5, 0.5, -1.0, 1.0);

		float x0 = -0.5f;
		float x1 = x0 + scale*sim->getIC().getNy()/static_cast<float>(sim->getIC().getNx());
		float y0 = -0.5f;
		float y1 = y0 + scale;
		float z  =  0.0f;

		glDisable(GL_DEPTH_TEST);
		glBegin(GL_QUADS);
		glColor4f(1.0f,1.0f,1.0f,0.2f);
		glTexCoord2f(0.0f, 0.0f); glVertex3f(x0, y0, z);
		glTexCoord2f(0.0f, 1.0f); glVertex3f(x1, y0, z);
		glTexCoord2f(1.0f, 1.0f); glVertex3f(x1, y1, z);
		glTexCoord2f(1.0f, 0.0f); glVertex3f(x0, y1, z);
		glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
		glEnd();
		glEnable(GL_DEPTH_TEST);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, 0);
		glDisable(GL_TEXTURE_2D);

		glMatrixMode(GL_PROJECTION);
		glPopMatrix();

		glMatrixMode(GL_MODELVIEW);
		glPopMatrix();

		glDeleteTextures(1, &tex);
	}
private:
	boost::shared_ptr<KPSimulator> sim;
};

#endif