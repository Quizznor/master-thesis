/**
  \file
  Iterator for particles from VShowerParticleList

  \author Lukas Nellen
  \version $Id$
  \date 4-Apr-2003
*/

#ifndef _utl_ShowerParticleIterator_h_
#define _utl_ShowerParticleIterator_h_

static const char CVSId_utl_ShowerParticleIterator[] =
  "$Id$";

// #include <iterator>

#include <CorsikaParticle.h>
#include <CorsikaShowerFileParticleIterator.h>

namespace io {

  /**
     \class ShowerParticleIterator ShowerParticleIterator.h
     "utl/ShowerParticleIterator.h"
     \brief Iterator to retrieve particles from utl::VShowerParticlList

     This class implements an input iterator to iterate over all
     particles in a utl::VShowerParticleList. The code doing the real
     work is in the implementation of the utl::VShowerParticleList
     interface.

     \note Since this is an input iterator, it only provides read
     access to the contents of the associated ShowerParticleList.

     \ingroup particles stl
  */

  class ShowerParticleIterator
    // : public std::iterator<std::input_iterator_tag, CorsikaParticle, ptrdiff_t>
  {

  public:
    explicit ShowerParticleIterator(CorsikaShowerFileParticleIterator* const fileIt = 0);

    CorsikaParticle& operator*();
    const CorsikaParticle& operator*() const;
    CorsikaParticle* operator->();
    const CorsikaParticle* operator->() const;

    /// Prefix increment
    ShowerParticleIterator& operator++();
    /// Postfix increment
    ShowerParticleIterator operator++(int);

    bool operator==(const ShowerParticleIterator& it) const;
    bool operator!=(const ShowerParticleIterator& it) const
    { return !operator==(it); }

  private:
    void CheckReference() const;

    CorsikaShowerFileParticleIterator* fParticles;
    CorsikaParticle* fCurrentParticle;
  public:
    int fCounter;
  };

}


#endif

// Configure (x)emacs for this file ...
// Local Variables:
// mode: c++
// compile-command: "cd ..; make -k"
// End:
