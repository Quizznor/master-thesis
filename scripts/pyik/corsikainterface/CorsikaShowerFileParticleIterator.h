/**
   \file
   Implementation of the VShowerFileParticleIterator for an Corsika generated
   shower file
   \author Troy Porter
   \author Lukas Nellen
   \version $Id$
   \date 22 May 2003
*/

#ifndef _io_CorsikaShowerFileParticleIterator_h_
#define _io_CorsikaShowerFileParticleIterator_h_

static const char CVSId__CorsikaShowerFileParticleIterator[] =
"$Id$";

#include <CorsikaBlock.h>
#include <RawCorsikaFile.h>
#include <CorsikaParticle.h>


namespace io {

  namespace Corsika {
    class RawFile;
  }

  class CorsikaParticle;


  /**
    \class CorsikaShowerFileParticleIterator
    \brief Implementation of the VShowerFileParticleIterator for an Corsika generated
           shower file
    \author Troy Porter
    \author Lukas Nellen
    \version $Id$
    \date 22 May 2003
    \ingroup corsika particles
  */

  class CorsikaShowerFileParticleIterator {

  public:
    typedef io::Corsika::RawFile::PositionType PositionType;

    CorsikaShowerFileParticleIterator(Corsika::RawFile& rawFile,
                                      const double timeOffset,
                                      const PositionType startPosition,
                                      const double version,
                                      const unsigned int observationLevel,
                                      const bool isThinned);

    virtual ~CorsikaShowerFileParticleIterator() { }

    virtual CorsikaParticle* GetOneParticle();
    virtual void Rewind();

  private:
    /// Low level reader of individual Corsika particles
    const io::Corsika::Block::ParticleData* GetOneParticleRecord();
    const io::Corsika::BlockUnthinned::ParticleData* GetOneParticleRecordUnthinned();

    Corsika::RawFile& fRawFile;
    const double fTimeOffset;
    const PositionType fStartPosition;
    const double fVersion;

    const unsigned int fObservationLevel;

    PositionType fCurrentPosition;
    int fParticleInBlock;
    bool fIteratorValid;

    CorsikaParticle fCurrentParticle;
    io::Corsika::Block fCurrentBlock;
    io::Corsika::BlockUnthinned fCurrentBlockUnthinned;

    bool fIsThinned;
    bool fBlockBufferValid;
  };

}


#endif // _io_CorsikaShowerFileParticleIterator_h_

// Configure (x)emacs for this file ...
// Local Variables:
// mode: c++
// compile-command: "make -C .. -k"
// End:
