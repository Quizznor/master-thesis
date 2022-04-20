/**
   \file
   Implementation of the Corsika shower reader

   \author Lukas Nellen
   \version $Id$
   \date 29 Jan 2004
*/

static const char CVSId[] =
  "$Id$";

#include <CorsikaShowerFile.h>
#include <CorsikaShowerFileParticleIterator.h>
#include <CorsikaIOException.h>
#include <RawCorsikaFile.h>
#include <CorsikaBlock.h>
#include <CorsikaShower.h>
#include <CorsikaUtilities.h>

#include <sstream>
#include <string>
#include <cmath>
#include <iostream>

using namespace std;
using namespace io;

static void log(const std::string& mess)
{
  std::cout << mess << std::endl;
}
static void log(const std::ostringstream& mess)
{
  std::cout << mess.str() << std::endl;
}

#define INFO(mess) log(mess);
#define ERROR(mess) log(mess);
#define FATAL(mess) log(mess);


CorsikaShowerFile::~CorsikaShowerFile()
{
}


CorsikaShowerFile::CorsikaShowerFile() :
  fRawFile(),
  fPositionToRaw(),
  fPositionOfEventTrailer(),
  fIdToPosition(),
  fCurrentPosition(0),
  fRunNumber(0),
  fObservationLevel(1),
  fIsThinned(true)
{
}


CorsikaShowerFile::CorsikaShowerFile(const std::string& theFileName,
                                     const bool requireParticleFile) :
  fRawFile(),
  fPositionToRaw(),
  fPositionOfEventTrailer(),
  fIdToPosition(),
  fCurrentPosition(0),
  fObservationLevel(1),
  fIsThinned(true)
{
  // only call Open() if the particle file is required (default behaviour)
  if (requireParticleFile) {
    Open(theFileName);
  } else {
    fLongFile = theFileName + ".long";  // particle file must be called "DATxxxxxx" here
    ostringstream info;
    info << "No particle file available, CORSIKA longitudinal file: " << fLongFile;
    INFO(info);
  }
}


void
CorsikaShowerFile::Open(const std::string& theFileName, bool scan)
{
  if (fRawFile.IsOpen())
    Close();

  // Compute the name for the long file
  string file = theFileName;

  if (file.find(".part") == string::npos)
    fLongFile = file + ".long";
  else
    fLongFile = file.replace(file.find(".part"), 5, ".long");

  ostringstream msg;
  msg << "CORSIKA longitudinal file: " << fLongFile;
  //INFO(msg);

  fRawFile.Open(theFileName);

  if (scan)
    ScanGroundFile();

  GotoPosition(0);
}


bool
CorsikaShowerFile::IsValid(const std::string& theFileName)
{
  CorsikaShowerFile file;
  file.Open(theFileName, false);
  return file.IsValid();
}

bool
CorsikaShowerFile::IsValid()
{
  Corsika::BlockUnthinned blockUnth;
  fRawFile.SeekTo(0);
  fRawFile.GetNextBlockUnthinned(blockUnth);
  if (!blockUnth.IsRunHeader()) {
    return false;
  }
  return true;
}

void
CorsikaShowerFile::ScanGroundFile()
{
  fRawFile.SeekTo(0);

  int eventsSoFar = 0;

  unsigned int blockIndex = 0;
  bool foundEventHeader = false;
  bool foundRunHeader = false;

  Corsika::BlockUnthinned blockUnth;
  while (fRawFile.GetNextBlockUnthinned(blockUnth) && !blockUnth.IsRunTrailer()) {
    ++blockIndex;
    if (blockUnth.IsEventHeader()) {
      fIsThinned = false;
      foundEventHeader = true;
      Corsika::RawFile::PositionType rawPosition = fRawFile.GetNextPosition();

      fPositionToRaw.push_back(rawPosition - 1);
      fIdToPosition[int(blockUnth.AsEventHeader().fEventNumber)] = eventsSoFar;

      ++eventsSoFar;
    } else
      if (blockUnth.IsEventTrailer())
        fPositionOfEventTrailer.push_back(fRawFile.GetNextPosition() - 1);
      else
        if (blockUnth.IsRunHeader()){
          foundRunHeader = true;
          fRunNumber = int(blockUnth.AsRunHeader().fRunNumber);
        }
    if ( blockIndex >400 && !foundRunHeader){
      string msg = "Error scanning Corsika ground file: "
        "could not find run header";
      ERROR(msg);
      throw io::CorsikaIOException(msg);
    }
    if ( blockIndex >400 && !foundEventHeader){
      break;
    }
    // adding break: assumption: only one event per file
    if ( foundEventHeader and foundRunHeader ) { break; }
  }
  //return; doesn't yield a valid result

  if (!blockUnth.IsRunTrailer()) {

    eventsSoFar = 0;
    blockIndex = 0;
    foundEventHeader = false;
    foundRunHeader = false;

    fRawFile.SeekTo(0);

    Corsika::Block blockTh;
    while (fRawFile.GetNextBlock(blockTh) && !blockTh.IsRunTrailer()) {
      ++blockIndex;
      if (blockTh.IsEventHeader()) {

        foundEventHeader = true;
        fIsThinned = true;
        Corsika::RawFile::PositionType rawPosition = fRawFile.GetNextPosition();

        fPositionToRaw.push_back(rawPosition - 1);
        fIdToPosition[int(blockTh.AsEventHeader().fEventNumber)] = eventsSoFar;

        ++eventsSoFar;
      } else if (blockTh.IsEventTrailer())
        fPositionOfEventTrailer.push_back(fRawFile.GetNextPosition() - 1);
      else if (blockTh.IsRunHeader()) {
        foundRunHeader = true;
        fRunNumber = int(blockTh.AsRunHeader().fRunNumber);
      }
      if (blockIndex > 400) {
        if (!foundRunHeader) {
          const string err = "Error scanning thinned Corsika ground file: "
            "could not find run header";
          ERROR(err);
          throw io::CorsikaIOException(err);
        }
        if (!foundEventHeader) {
          const string err = "Error scanning Corsika ground file: "
            "could not find Event header";
          ERROR(err);
          throw io::CorsikaIOException(err);
        }
      }
      if (foundEventHeader && foundRunHeader) { break; }
    }
    /*
    if (!blockTh.IsRunTrailer()) {
      const string err = "Error scanning Corsika ground file: could not find run end";
      ERROR(err);
      throw io::CorsikaIOException(err);
    }
    */
  }
  /*
  if (fPositionToRaw.size() != fPositionOfEventTrailer.size()) {
    const string err = "Found different number of event-headers and -trailers";
    ERROR(err);
    throw io::CorsikaIOException(err);
  }
  */
}


void
CorsikaShowerFile::Close()
{
  if (fRawFile.IsOpen())
    fRawFile.Close();

  fPositionToRaw.clear();
  fIdToPosition.clear();
}


Status
CorsikaShowerFile::Read()
{
  if (!fRawFile.IsOpen() || fCurrentPosition >= fPositionToRaw.size())
    return eEOF;

  fRawFile.SeekTo(fPositionToRaw[fCurrentPosition]);

  if (fIsThinned) {
    Corsika::Block headerBlock;
    if (!fRawFile.GetNextBlock(headerBlock)) {
      ostringstream err;
      err << "Cannot read CORSIKA shower header for position "
          << fCurrentPosition;
      FATAL(err);
      return eFail;
    }

    if (!headerBlock.IsEventHeader()) {
      ostringstream err;
      err << "First block at position " << fCurrentPosition
          << " is not event header";
      FATAL(err);
      return eFail;
    }
    const Corsika::Block::EventHeader& header = headerBlock.AsEventHeader();

/*
    fRawFile.SeekTo(fPositionOfEventTrailer[fCurrentPosition]);

    Corsika::Block trailerBlock;
    if (!fRawFile.GetNextBlock(trailerBlock)) {
      ostringstream err;
      err << "Cannot read CORSIKA shower trailer for position "
          << fCurrentPosition;
      FATAL(err);
      return eFail;
    }
    if (!trailerBlock.IsEventTrailer()) {
      ostringstream err;
      err << "Block at position " << fCurrentPosition
          << " is not event trailer";
      FATAL(err);
      return eFail;
    }
*/
    //const Corsika::Block::EventTrailer& trailer = trailerBlock.AsEventTrailer();
    const Corsika::Block::EventTrailer& trailer = Corsika::Block::EventTrailer();

    if (fObservationLevel > header.fObservationLevels) {
      ostringstream info;
      info << "The requested observation level: " << fObservationLevel
           << " does not exist (max obs. level: "
           << header.fObservationLevels << "), "
           << "switching to level 1.";
      fObservationLevel = 1;
      INFO(info);
    }

    #warning add units here
    // Corsika starts at the top of the atmosphere, not
    const float heightObsLevel =
      header.fObservationHeight[int(header.fObservationLevels) - 1]; // in cm
    const float heightFirstInt = abs(header.fZFirst); // in cm

    #warning add units here
    double hReference;
    const double hAtmBoundary = 112.8292*1e5; // in cm

    // for the SLANT and CURVED options, clock starts at the margin of
    // the atmosphere. This is indicated by fZFirst < 0
    if (header.fZFirst < 0.) {
      hReference = hAtmBoundary;
    } else {
      hReference = heightFirstInt;
    }

    double timeShift = 0;

    const double Zenith = header.fTheta;
    const double cosZenith = cos(Zenith);

    #warning add units here
    const double kSpeedOfLight = 299792458*1e2; // speed of light in cm/s

    if (header.fFlagCurved) {

      INFO("CURVED version");

     if (Corsika::CorsikaToPDG(int(header.fParticleId)) == CorsikaParticle::ePhoton)
        hReference = heightFirstInt;

      // value taken from CORSIKA
      const double kREarth = 6.371315e8; // radius of the earth in cm

      timeShift = (pow((kREarth + heightObsLevel)*cosZenith, 2) +
                   pow(hReference - heightObsLevel, 2) +
                   2*(kREarth + heightObsLevel)*(hReference - heightObsLevel));
      timeShift = sqrt(timeShift);
      timeShift -= (kREarth + heightObsLevel)*cosZenith;
      timeShift /= kSpeedOfLight;

      ostringstream info;
      info << "TimeShift to core: " << timeShift/1e9; // output in ns
      INFO(info);

    } else
      timeShift = (hReference - heightObsLevel) / (cosZenith * kSpeedOfLight);

/*
    CorsikaShowerFileParticleIterator* particleIterator = new CorsikaShowerFileParticleIterator(fRawFile,
                                                       timeShift,
                                                       fPositionToRaw[fCurrentPosition] + 1,
                                                       header.fVersion,
                                                       fObservationLevel,
                                                       fIsThinned);
*/

    // setting this pointer to NULL, so that particle iteration can't be done
    CorsikaShowerFileParticleIterator* particleIterator = NULL;
    fCurrentShower = CorsikaShower(header, trailer, particleIterator);

    ++fCurrentPosition;

    return eSuccess;
  }

  return eFail;
}


Status
CorsikaShowerFile::FindEvent(const unsigned int eventId)
{
  const IdToPositionMap::const_iterator iter = fIdToPosition.find(eventId);
  if (iter == fIdToPosition.end())
    return eFail;
  else
    return GotoPosition(iter->second);
}


Status
CorsikaShowerFile::GotoPosition(const unsigned int position)
{
  if (position >= fPositionToRaw.size())
    return eFail;

  fCurrentPosition = position;
  return eSuccess;
}


int
CorsikaShowerFile::GetNEvents()
{
  if (fRawFile.IsOpen())
    return fIdToPosition.size();

  const string msg = "Cannot request number of events from closed file";
  ERROR(msg);
  throw CorsikaIOException(msg);
}


// Configure (x)emacs for this file ...
// Local Variables:
// mode: c++
// compile-command: "make -C .. -k"
// End:
