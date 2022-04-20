/**
   \file
   

   \author Javier Gonzalez
   \version $Id: TEMPLATE.cc.tpl,v 1.4 2003/09/25 14:38:19 lukas Exp $
   \date 04 May 2011
*/

static const char CVSId[] =
"$Id$";


#include <CorsikaShower.h>
#include <CorsikaUtilities.h>
#include <CorsikaBlock.h>
#include <iostream>

using namespace io;
using namespace std;

CorsikaShower::CorsikaShower(const Corsika::Block::EventHeader& header, const Corsika::Block::EventTrailer& trailer, CorsikaShowerFileParticleIterator* particleIt):
  fPrimaryParticle(Corsika::CorsikaToPDG(int(header.fParticleId))),
  fEnergy(header.fEnergy),
  fZFirst(header.fZFirst),
  fZenith(header.fTheta),
  fAzimuth(header.fPhi),
  fMuonNumber(0),
  fMinRadiusCut(header.fRMaxThinning),
  fShowerNumber(int(header.fEventNumber)),
  fShowerRunId(int(header.fRunNumber)),
  fEMEnergyCutoff(header.fCutoffElectrons),
  fMuonEnergyCutoff(header.fCutoffMuons),
  fParticleIterator(particleIt),
  fParticles(fParticleIterator)
{
}

// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "make -C .. -k"
// End:
