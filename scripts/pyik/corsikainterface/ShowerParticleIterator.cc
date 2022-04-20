/**
  \file
  Implementation for ShowerParticleIterator

  \author Lukas Nellen
  \date 8-apr-2003
  \version $Id$
*/

#include <ShowerParticleIterator.h>
#include <CorsikaIOException.h>
#include <iostream>


using namespace std;

namespace io {

  ShowerParticleIterator::
  ShowerParticleIterator(CorsikaShowerFileParticleIterator* const fileIt) :
    fParticles(fileIt),
    fCurrentParticle(0),
    fCounter(0)
  {
    if (fParticles) {
      fParticles->Rewind();
      fCurrentParticle = fParticles->GetOneParticle();
      if (!fCurrentParticle)
        fParticles = 0;
    }
  }


  void
  ShowerParticleIterator::
  CheckReference()
    const
  {
    if (!fCurrentParticle)
      throw CorsikaIOException("Dereferencing invalid ShowerParticleIterator");
  }


  CorsikaParticle&
  ShowerParticleIterator::
  operator*()
  {
    CheckReference();
    return *fCurrentParticle;
  }


  const CorsikaParticle&
  ShowerParticleIterator::
  operator*()
    const
  {
    CheckReference();
    return *fCurrentParticle;
  }


  CorsikaParticle*
  ShowerParticleIterator::
  operator->()
  {
    CheckReference();
    return fCurrentParticle;
  }


  const CorsikaParticle*
  ShowerParticleIterator::
  operator->()
    const
  {
    CheckReference();
    return fCurrentParticle;
  }


  ShowerParticleIterator&
  ShowerParticleIterator::
  operator++()
  {
    if (!fParticles)
      throw CorsikaIOException("Incrementing invalid ShowerParticleIterator");
    fCurrentParticle = fParticles->GetOneParticle();
    if (!fCurrentParticle)
      fParticles = 0;
    ++fCounter;
    return *this;
  }


  ShowerParticleIterator
  ShowerParticleIterator::
  operator++(int)
  {
    ShowerParticleIterator old(*this);
    ++(*this);
    ++fCounter;
    return old;
  }


  bool
  ShowerParticleIterator::
  operator==(const ShowerParticleIterator& it)
    const
  {
    return fParticles == it.fParticles &&
      fCurrentParticle == it.fCurrentParticle;
  }
}

// Configure (x)emacs for this file ...
// Local Variables:
// mode: c++
// compile-command: "cd ..; make -k"
// End:
