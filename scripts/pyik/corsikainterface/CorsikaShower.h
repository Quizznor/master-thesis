/**
   \file
   

   \author Javier Gonzalez
   \version $Id: TEMPLATE.h.tpl,v 1.5 2003/09/25 14:38:19 lukas Exp $
   \date 04 May 2011
*/

#ifndef _io_CorsikaShower_h_
#define _io_CorsikaShower_h_

static const char CVSId_io_CorsikaShower[] =
"$Id$";

#include <CorsikaBlock.h>
#include <CorsikaShowerFileParticleIterator.h>
#include <ShowerParticleList.h>

namespace io {

  class CorsikaShower {
  public:
    CorsikaShower():
      fParticleIterator(0),
      fParticles(fParticleIterator)
    {}
    CorsikaShower(const Corsika::Block::EventHeader& , const Corsika::Block::EventTrailer& trailer, CorsikaShowerFileParticleIterator* particleIt);
    ~CorsikaShower(){}

    int GetPrimary() const            {return fPrimaryParticle;   }
    float GetEnergy() const           {return fEnergy;         }
    float GetZFirst() const           {return fZFirst;         }
    float GetMuonNumber() const       {return fMuonNumber;      }
    float GetZenith() const           {return fZenith;          }
    float GetAzimuth() const          {return fAzimuth;         }
    float GetMinRadiusCut() const     {return fMinRadiusCut;    }
    int GetShowerNumber() const       {return fShowerNumber;    }
    int GetShowerRunId() const        {return fShowerRunId;     }
    float GetEMEnergyCutoff() const   {return fEMEnergyCutoff;  }
    float GetMuonEnergyCutoff() const {return fMuonEnergyCutoff;}

    ShowerParticleList<CorsikaShowerFileParticleIterator>& GetParticles()
    {return fParticles;}

  private:
    int fPrimaryParticle;
    float fEnergy;
    float fZFirst;
    float fMuonNumber;
    float fZenith;
    float fAzimuth;
    float fMinRadiusCut;
    int fShowerNumber;
    int fShowerRunId;
    float fEMEnergyCutoff;
    float fMuonEnergyCutoff;

    CorsikaShowerFileParticleIterator* fParticleIterator;
    ShowerParticleList<CorsikaShowerFileParticleIterator> fParticles;
  };

} // io


#endif // _io_CorsikaShower_h_

// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "make -C .. -k"
// End:
