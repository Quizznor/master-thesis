/**
   \file
   Reader for Corsika generated shower file

   \author Troy Porter, Lukas Nellen
   \version $Id$
   \date 07 Jul 2003, 29 Jan 2004
*/

#ifndef _io_CorsikaShowerFile_h_
#define _io_CorsikaShowerFile_h_

static const char CVSId_io_CorsikaShowerFile[] =
"$Id$";

#include <RawCorsikaFile.h>
#include <CorsikaShowerFileParticleIterator.h>
#include <CorsikaIOException.h>
#include <CorsikaShower.h>

#include <string>
#include <map>
#include <vector>


namespace io {

  enum Status {
    eSuccess,
    eFail,
    eEOF
  };

  /**
     \class CorsikaShowerFile CorsikaShowerFile.h "io/CorsikaShowerFile.h"

     \brief Read data from the output of CORSIKA

     \author Lukas Nellen
     \date 29 Jan 2004
     \ingroup corsika
  */
  class CorsikaShowerFile {

  public:

    CorsikaShowerFile();
    CorsikaShowerFile(const std::string& theFileName, const bool requireParticleFile = true);

    virtual ~CorsikaShowerFile();

    virtual void Open(const std::string& theFileName, bool scan = true);

    virtual void Close();

    virtual Status Read();

    virtual Status FindEvent(unsigned int eventId);

    virtual Status GotoPosition(unsigned int position);

    virtual int GetNEvents();

    bool IsValid();

    static bool IsValid(const std::string& theFileName);

    const CorsikaShower& GetCurrentShower() const {return fCurrentShower;}
    CorsikaShower& GetCurrentShower() {return fCurrentShower;}

  private:
    typedef std::vector<io::Corsika::RawFile::PositionType> PositionVector;
    typedef std::map<unsigned int, unsigned int> IdToPositionMap;

    /// Collect information about ground file
    void ScanGroundFile();

    CorsikaShower fCurrentShower;

    io::Corsika::RawFile fRawFile;
    std::string fLongFile;
    PositionVector fPositionToRaw;
    PositionVector fPositionOfEventTrailer;
    IdToPositionMap fIdToPosition;
    unsigned int fCurrentPosition;
    int fRunNumber;
    unsigned int fObservationLevel;
    bool fIsThinned;

  };

}


#endif

// Configure (x)emacs for this file ...
// Local Variables:
// mode: c++
// compile-command: "make -C .. -k"
// End:
