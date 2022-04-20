/**
   \file
   Implementation file for CorsikaShowerFileParticleIterator class
   \author Troy Porter
   \version $Id$
   \date 22 May 2003
*/

#include "CorsikaShowerFileParticleIterator.h"

#include "RawCorsikaFile.h"
#include "CorsikaBlock.h"
#include "CorsikaIOException.h"
#include "CorsikaUtilities.h"
#include "CorsikaParticle.h"

#include <iostream>
#include <sstream>
#include <cmath>

using namespace io;
using ::io::Corsika::Block;
using ::io::Corsika::BlockUnthinned;

#define ERROR(mess) std::cerr << mess << std::endl;
#define INFO(mess) std::cout << mess << std::endl;

using namespace std;

CorsikaShowerFileParticleIterator::
CorsikaShowerFileParticleIterator(Corsika::RawFile& rawFile,
                                  const double timeOffset,
                                  const PositionType startPosition,
                                  const double version,
                                  const unsigned int observationLevel,
                                  const bool isThinned) :
  fRawFile(rawFile),
  fTimeOffset(timeOffset),
  fStartPosition(startPosition),
  fVersion(version),
  fObservationLevel(observationLevel),
  fCurrentPosition(0),
  fParticleInBlock(0),
  fIteratorValid(false),
  fCurrentParticle(),
  fCurrentBlock(),
  fCurrentBlockUnthinned(),
  fIsThinned(isThinned),
  fBlockBufferValid(false)
{ }


void
CorsikaShowerFileParticleIterator::Rewind()
{
  fCurrentPosition = fStartPosition;
  fParticleInBlock = 0;
  fBlockBufferValid = false;
  fIteratorValid = true;
}


CorsikaParticle*
CorsikaShowerFileParticleIterator::GetOneParticle()
{

  for (;;) {

    if (fIsThinned) {
      const Block::ParticleData* const corsikaParticle = GetOneParticleRecord();

      // end of particle list
      if (!corsikaParticle)
        return 0;

      const int particleId =
        Corsika::CorsikaToPDG(int(corsikaParticle->fDescription/1000));

      // skip unknown particles...
      if (particleId == CorsikaParticle::eUndefined)
        continue;

      const short unsigned int obsLevel =  fmod(corsikaParticle->fDescription, 10);

      if (obsLevel != fObservationLevel)
        continue;

      CorsikaParticle p = CorsikaParticle(corsikaParticle);

      fCurrentParticle = p;

      return &fCurrentParticle;

    } else {

      const BlockUnthinned::ParticleData* const corsikaParticle = GetOneParticleRecordUnthinned();

      // end of particle list
      if (!corsikaParticle)
        return 0;

      const int particleId =
        Corsika::CorsikaToPDG(int(corsikaParticle->fDescription/1000));

      // skip unknown particles...
      if (particleId == CorsikaParticle::eUndefined)
        continue;

      const short unsigned int obsLevel =  fmod(corsikaParticle->fDescription, 10);

      if (obsLevel != fObservationLevel)
        continue;

      fCurrentParticle = CorsikaParticle(corsikaParticle);

      return &fCurrentParticle;

    }
  }
}


/**
   Get one Corsika particle record from current event.

   Returns a pointer to the current corsika particle data
   sub-block. It doesn't check for the type of sub-block. It is the
   responsibility of the caller to ignore unwanted records,
   e.g. particle unkown to the client.

   Returns null pointer when reading beyond the end of the particle
   records. This should only happen when the event-trailer is reached.

   \note This is coded for the ground particle information only.

   \note The current implementation cannot deal with logitudinal
   particle records in the data stream.
 */
const Block::ParticleData*
CorsikaShowerFileParticleIterator::GetOneParticleRecord()
{
  using std::ostringstream;

  if (!fIteratorValid)
    throw CorsikaIOException("CorsikaShowerFileParticleIterator not valid.");

  if (!fBlockBufferValid) {
    fRawFile.SeekTo(fCurrentPosition);
    if (!fRawFile.GetNextBlock(fCurrentBlock)) {
      ostringstream msg;
      msg << "Error reading block " << fCurrentPosition << " in CORSIKA file.";
      ERROR(msg);
      throw CorsikaIOException(msg.str());
    }

    if (fCurrentBlock.IsControl()) { // end of particle records
      fIteratorValid = false;
      return 0;
    }
  }

  const Block::ParticleData* const currentRecord =
    fCurrentBlock.AsParticleBlock().fParticle + fParticleInBlock;
  ++fParticleInBlock;

  if (fParticleInBlock >= Block::kParticlesInBlock) {
    ++fCurrentPosition;
    fParticleInBlock = 0;
    fBlockBufferValid = false;
  }

  return currentRecord;
}


const BlockUnthinned::ParticleData*
CorsikaShowerFileParticleIterator::GetOneParticleRecordUnthinned()
{
  using std::ostringstream;

  if (!fIteratorValid)
    throw CorsikaIOException("CorsikaShowerFileParticleIterator not valid.");

  if (!fBlockBufferValid) {
    fRawFile.SeekTo(fCurrentPosition);
    if (!fRawFile.GetNextBlockUnthinned(fCurrentBlockUnthinned)) {
      ostringstream msg;
      msg << "Error reading block " << fCurrentPosition << " in CORSIKA file.";
      ERROR(msg);
      throw CorsikaIOException(msg.str());
    }

    if (fCurrentBlockUnthinned.IsControl()) { // end of particle records
      fIteratorValid = false;
      return 0;
    }
  }

  const BlockUnthinned::ParticleData* const currentRecord =
    fCurrentBlockUnthinned.AsParticleBlock().fParticle + fParticleInBlock;
  ++fParticleInBlock;

  if (fParticleInBlock >= BlockUnthinned::kParticlesInBlock) {
    ++fCurrentPosition;
    fParticleInBlock = 0;
    fBlockBufferValid = false;
  }

  return currentRecord;
}




// Configure (x)emacs for this file ...
// Local Variables:
// mode: c++
// compile-command: "make -C .. -k"
// End:
