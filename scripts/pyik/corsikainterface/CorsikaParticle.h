/**
   \file


   \author Javier Gonzalez
   \version $Id: TEMPLATE.h.tpl,v 1.5 2003/09/25 14:38:19 lukas Exp $
   \date 04 May 2011
*/

#ifndef __CorsikaParticle_h_
#define __CorsikaParticle_h_

static const char CVSId__CorsikaParticle[] =
"$Id$";


#include "CorsikaBlock.h"


namespace io {


    struct ParticleData;

    class CorsikaParticle {
    public:

      enum Type {
        eUndefined = 0,
        eElectron = 11, ePositron = -11,
        eNuElectron = 12, eAntiNuElectron = -12,
        eMuon = 13, eAntiMuon = -13,
        eNuMuon = 14, eAntiNuMuon = -14,
        eTau = 15, eAntiTau = -15,
        eNuTau = 16, eAntiNuTau = -16,
        ePhoton = 22,
        ePiZero = 111,
        ePiPlus = 211, ePiMinus = -211,
        eEta = 221,
        eKaon0L = 130, eKaon0S = 310,
        eKaonPlus = 321, eKaonMinus = -321,
        eLambda = 3122, eAntiLambda = -3122,
        eLambdac = 4122,
        eNeutron = 2112, eAntiNeutron = -2112,
        eProton = 2212, eAntiProton = -2212,
        // Selected nuclei.
        // Note: This is an inconsistency left in for hysterical raisins only.
        //       Cf. utl::NucleusProperties instead!
        //       Usage example thereof can be found in CONEXFile.cc.
        eIron = 1000026056
      };

      CorsikaParticle();
      CorsikaParticle(const Corsika::Block::ParticleData* particle);
      CorsikaParticle(const Corsika::BlockUnthinned::ParticleData* particle);
      ~CorsikaParticle();


      bool IsParticle() const
      { return 0 < fDescription && fDescription < 100000; }
      bool IsNucleus() const
      { return 100000 <= fDescription && fDescription < 9900000; }
      bool IsCherenkov() const
      { return 9900000 <= fDescription; }

      float fDescription;     // Particle ID
      float fPx, fPy, fPz;    // GeV
      float fX, fY;           // cm
      float fTorZ;            // ns or cm
      float fWeight;          // absent if CORSIKA compiled w/o thinning

      /// Calculate particle type code for a given (A, Z)
      static int NucleusCode(const int theCharge, const int theAtomicNumber);

    }; // CorsikaParticle

} //


#endif // __CorsikaParticle_h_

// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "make -C .. -k"
// End:
