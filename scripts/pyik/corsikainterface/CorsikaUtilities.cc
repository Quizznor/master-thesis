/**
   \file
   Converter from CORSIKA to PDG particle codes.

   \author Lukas Nellen
   \version $Id$
   \date 21 Apr 2004
*/

static const char CVSId[] =
"$Id$";


#include "CorsikaUtilities.h"
#include "CorsikaIOException.h"
#include "CorsikaParticle.h"

#include <iostream>
#include <map>
using std::map;
#include <utility>
#include <sstream>
using std::ostringstream;

#define ERROR(mess) std::cerr << mess << std::endl;

namespace {
  map<int, int> corsikaToPDGMap;

  void InsertCorsikaToPDG(int theCorsikaCode, int thePDGCode)
  {
    if ( !corsikaToPDGMap.insert(std::make_pair(theCorsikaCode,
                                                thePDGCode)).second ) {
      ostringstream msg;
      msg << "Cannot insert pair ("
          << theCorsikaCode << ", " << thePDGCode
          << ") into CorsikaToPDG map.";
      ERROR(msg);
      throw io::CorsikaIOException(msg.str());
    }
  }

  void InitCorsikaToPDGMap()
  {
    InsertCorsikaToPDG(1, io::CorsikaParticle::ePhoton);
    InsertCorsikaToPDG(2, io::CorsikaParticle::ePositron);
    InsertCorsikaToPDG(3, io::CorsikaParticle::eElectron);

    InsertCorsikaToPDG(5, io::CorsikaParticle::eAntiMuon);
    InsertCorsikaToPDG(6, io::CorsikaParticle::eMuon);
    InsertCorsikaToPDG(7, io::CorsikaParticle::ePiZero);
    InsertCorsikaToPDG(8, io::CorsikaParticle::ePiPlus);
    InsertCorsikaToPDG(9, io::CorsikaParticle::ePiMinus);

    InsertCorsikaToPDG(10, io::CorsikaParticle::eKaon0L);
    InsertCorsikaToPDG(11, io::CorsikaParticle::eKaonPlus);
    InsertCorsikaToPDG(12, io::CorsikaParticle::eKaonMinus);
    InsertCorsikaToPDG(13, io::CorsikaParticle::eNeutron);
    InsertCorsikaToPDG(14, io::CorsikaParticle::eProton);
    InsertCorsikaToPDG(15, io::CorsikaParticle::eAntiProton);
    InsertCorsikaToPDG(16, io::CorsikaParticle::eKaon0S);
    InsertCorsikaToPDG(17, io::CorsikaParticle::eEta);
    InsertCorsikaToPDG(18, io::CorsikaParticle::eLambda);

    InsertCorsikaToPDG(25, io::CorsikaParticle::eAntiNeutron);
    InsertCorsikaToPDG(26, io::CorsikaParticle::eAntiLambda);

    InsertCorsikaToPDG(66, io::CorsikaParticle::eNuElectron);
    InsertCorsikaToPDG(67, io::CorsikaParticle::eAntiNuElectron);
    InsertCorsikaToPDG(68, io::CorsikaParticle::eNuMuon);
    InsertCorsikaToPDG(69, io::CorsikaParticle::eAntiNuMuon);

    InsertCorsikaToPDG(71, io::CorsikaParticle::eEta);
    InsertCorsikaToPDG(72, io::CorsikaParticle::eEta);
    InsertCorsikaToPDG(73, io::CorsikaParticle::eEta);
    InsertCorsikaToPDG(74, io::CorsikaParticle::eEta);
  }

} // namespace

int
io::Corsika::CorsikaToPDG(int theCorsikaCode)
{
  if (theCorsikaCode < 100) {

    if (!corsikaToPDGMap.size())
      InitCorsikaToPDGMap();

    map<int, int>::const_iterator index =
      corsikaToPDGMap.find(theCorsikaCode);

    if (index != corsikaToPDGMap.end()) {
      return index->second;
    } else {
      return CorsikaParticle::eUndefined;
    }

  } else if (theCorsikaCode < 9900) {                   // nucleus

    unsigned int Z = theCorsikaCode % 100;
    unsigned int A = theCorsikaCode / 100;
    return CorsikaParticle::NucleusCode(Z, A);

  } else {                                              // Cherenkov
    return CorsikaParticle::eUndefined;
  }
} // CorsikaToPDG

/** Rotate form the Corsika coordinate system to Auger standard.
    Auger places the x axis east and the y axis north. Corsika places
    the x axis in the magnetic north and the y axis west. Therefore,
    the geomagnetic field direction for the location and time of an
    event has to be taken into account for the correct transformation.

    Auger uses the incomming direction, Corsika the outgoing
    direction. This adds 180 deg to the difference.

    \note The current value for the magnetic field rotation is valid
    for the southern site in 2002. The drift is around 2.5 degree in
    20 years.
*/
// double
// io::Corsika::CorsikaAzimuthToAuger(const double corsikaAzimuth)
// {
//   using utl::deg;
//   using utl::GeometryUtilities::NormalizeAngle;
//   const double kMagneticFieldDeclination = 4.233*deg;

// #warning Uses const. declination of mag. field in conversion of Corsika coord
//   //  return NormalizeAngle(corsikaAzimuth - 90*deg - kMagneticFieldDeclination);
//   return NormalizeAngle(corsikaAzimuth + 90.0*deg - kMagneticFieldDeclination);
// }


// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "make -C .. -k"
// End:
