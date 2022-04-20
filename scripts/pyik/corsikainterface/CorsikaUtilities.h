
#ifndef _io_CorsikaToPDG_h_
#define _io_CorsikaToPDG_h_

/**
   \file

   \brief converters from CORSIKA to PDG particle codes
   \author Lukas Nellen
   \version $Id$
   \date 21 Apr 2004
   \ingroup corsika particles
*/

static const char CVSId_io_CorsikaToPDG[] =
"$Id$";

namespace io {

  namespace Corsika {

    /// Convert Corsika particle code to PDG
    int CorsikaToPDG(int theCorsikaCode);

    /// Returns the azimuth rotated from Corisika's system to Auger standard
    // double CorsikaAzimuthToAuger(const double corsikaAzimuth);

  } // namespace Corsika

} // io


#endif // _io_CorsikaToPDG_h_

// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "make -C .. -k"
// End:
