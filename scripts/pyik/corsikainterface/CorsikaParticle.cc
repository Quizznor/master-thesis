/**
   \file
   

   \author Javier Gonzalez
   \version $Id: TEMPLATE.cc.tpl,v 1.4 2003/09/25 14:38:19 lukas Exp $
   \date 04 May 2011
*/

static const char CVSId[] =
"$Id$";


#include "CorsikaParticle.h"
#include "CorsikaBlock.h"
#include "NucleusProperties.h"
#include <iostream>


using namespace io;
using namespace std;

CorsikaParticle::CorsikaParticle():
  fDescription(0),
  fPx(0),
  fPy(0),
  fPz(0),
  fX(0),
  fY(0),
  fTorZ(0),
  fWeight(0)
{
}


CorsikaParticle::CorsikaParticle(const Corsika::Block::ParticleData* particle):
  fDescription(particle->fDescription),
  fPx(particle->fPx),
  fPy(particle->fPy),
  fPz(particle->fPz),
  fX(particle->fX),
  fY(particle->fY),
  fTorZ(particle->fTorZ),
  fWeight(particle->fWeight)
{
}


CorsikaParticle::CorsikaParticle(const Corsika::BlockUnthinned::ParticleData* particle):
  fDescription(particle->fDescription),
  fPx(particle->fPx),
  fPy(particle->fPy),
  fPz(particle->fPz),
  fX(particle->fX),
  fY(particle->fY),
  fTorZ(particle->fTorZ),
  fWeight(1)
{
}


CorsikaParticle::~CorsikaParticle()
{
}


int
CorsikaParticle::NucleusCode(const int theCharge, const int theAtomicNumber)
{
  return utl::NucleusProperties::TypeCode(theCharge, theAtomicNumber);
}

// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "make -C .. -k"
// End:
