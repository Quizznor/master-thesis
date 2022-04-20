/**
   \file
   Interface for particles lists in a shower file from simulation
   \author Lukas Nellen
   \version $Id$
   \date 4-Apr-2003
*/

#ifndef _utl_ShowerParticleList_h_
#define _utl_ShowerParticleList_h_

static const char CVSId_utl_ShowerParticleList[] =
"$Id$";

#include <ShowerParticleIterator.h>

namespace io {

  /**
     \class ShowerParticleList ShowerParticleList.h "utl/ShowerParticleList.h"

     \brief Interface class for accessing a list of particles from a
     shower file
  */
  template<class VShowerFileParticleIterator>
  class ShowerParticleList
  {
  public:
    ShowerParticleList(VShowerFileParticleIterator* theInterface = 0) :
      fFileInterface(theInterface)
    {
    }

    virtual ~ShowerParticleList() { }

    // ShowerParticleList& operator=(const ShowerParticleList& theList)
    // {
    //   if (this != &theList)
    //     fFileInterface = theList.fFileInterface;
    //   return *this;
    // }

    ShowerParticleIterator begin() const
    {
      return ShowerParticleIterator(fFileInterface);
    }

    ShowerParticleIterator end() const
    {
      return ShowerParticleIterator(0);
    }

  private:
    VShowerFileParticleIterator* fFileInterface;
  };

}

#endif

// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "cd ..; make -k"
// End:
