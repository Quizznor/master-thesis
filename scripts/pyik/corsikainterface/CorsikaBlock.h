/**
   \file
   On-disk block of raw CORSIKA files

   \note This is an internal data structure for the Corsika reader
   that is not installed with the rest of the framework.

   \author Lukas Nellen
   \version $Id$
   \date 19 Nov 2003
*/

#ifndef _io_Corsika_Block_h_
#define _io_Corsika_Block_h_

static const char CVSId_io_Corsika_Block[] =
  "$Id$";

#include <cstddef>


namespace io {

  namespace Corsika {

    /**
       \class Block CorsikaBlock.h "Corsika/CorsikaBlock.h"

       \brief This class represents a corsika block. It deals with all
       the different sub-types of blocks. Grouping of Blocks into a
       block on disk is done by the io::Corsika::RawFile class.

       \note This type deals with the machine-dependent representation
       on-disk. This class has to be reviewed when porting to different
       architectures or compilers.

       \todo Implement support to test padding requirements when
       configuring the software. Alternatively, think of a way to
       determine the size of a corsika block with padding at run-time
       and use C++'s placement new to deal with padding offsets.

       \todo Deal with byte-ordering issues to makes showers
       transferable between MSB and LSB architectures.

       \author Lukas Nellen
       \date 19 Nov 2003
       \ingroup corsika
    */
    class Block {
    public:
      typedef float RWORD;      // 4 byte real
      //typedef int   IWORD;    // 4 byte int

      static const int kMaxObservationLevels = 10;
      static const int kParticlesInBlock  = 39;

      /// Padding for thinning (in RWORDS): 39 (thinning) or 0 (no thinning)
      static const int kPaddingForThinning = 39;

      /*!
        \class BlockID
        \brief Sub-block used in CORSIKA files
      */
      class BlockID {
      public:
        /// Length of sub-block identifier
        static const size_t kLength = 4;

        /// set from c-string (for testing)
        void SetID(const char* const theID);

        /// Compare ID's
        bool Is(const char* const theID) const;

      private:
        char fID[kLength];      // hold strings like RUNH etc to
                                // identify the block type
      }; // BlockID

      /*!
        \struct RunHeader
        \brief run header struct for Corsika files
      */
      struct RunHeader {
        BlockID fID;
        RWORD fRunNumber;
        RWORD fDateStart;
        RWORD fVersion;

        RWORD fObservationLevels;
        RWORD fObservationHeight[kMaxObservationLevels]; // in cm

        RWORD fSpectralSlope;
        RWORD fEMin, fEMax;     // in GeV

        RWORD fFlagEGS4;
        RWORD fFlagNKG;

        RWORD fCutoffHadrons;   // in GeV
        RWORD fCutoffMuons;     // in GeV
        RWORD fCutoffElectrons; // in GeV
        RWORD fCutoffPhotons;   // in GeV

        RWORD fConstC[50];
        RWORD fConstCC[20];
        RWORD fConstCKA[40];
        RWORD fConstCETA[5];
        RWORD fConstCSTRBA[11];
        RWORD fConstUNUSED[4];
        RWORD fConstCAN[50];
        RWORD fConstCANN[50];
        RWORD fConstAATM[5];
        RWORD fConstBATM[5];
        RWORD fConstCATM[5];
        RWORD fConstNFLAIN;
        RWORD fConstNFLDIF;
        RWORD fConstNFLPI;
        RWORD fConstNFLCHE;

        RWORD fPad[kPaddingForThinning];
      }; // RunHeader

      /*!
        \struct EventHeader
        \brief event header struct for Corsika files
      */
      struct EventHeader {
        BlockID fID;
        RWORD fEventNumber;
        RWORD fParticleId;
        RWORD fEnergy;
        RWORD fStartingAltitude;        // g cm^-2
        RWORD fFirstTarget;     //
        RWORD fZFirst;          // cm
        RWORD fPx, fPy, fPz;    // GeV
        RWORD fTheta;           // zenith in radians
        RWORD fPhi;             // azimuth in radians

        RWORD fRandomSequences;
        struct {
          RWORD fSeed;
          RWORD fInitialCallsMod;
          RWORD fInitialCallsDiv;
        } fSeeds[10];

        RWORD fRunNumber;
        RWORD fDateStart;
        RWORD fVersion;

        RWORD fObservationLevels;
        RWORD fObservationHeight[kMaxObservationLevels]; // in cm

        RWORD fSpectralSlope;   //
        RWORD fEMin, fEMax;     // in GeV

        RWORD fCutoffHadrons;   // in GeV
        RWORD fCutoffMuons;     // in GeV
        RWORD fCutoffElectrons; // in GeV
        RWORD fCutoffPhotons;   // in GeV

        RWORD fNFLAIN;
        RWORD fNFLDIF;
        RWORD fNFLPI0;
        RWORD fNFLPIF;
        RWORD fNFLCHE;
        RWORD fNFRAGM;

        RWORD fBx, fBz;         // magnetic field in mu T
        RWORD fFlagEGS4;
        RWORD fFlagNKG;

        RWORD fFlagGeisha;
        RWORD fFlagVenus;
        RWORD fFlagCerenkov;
        RWORD fFlagNeutrino;
        RWORD fFlagCurved;
        RWORD fFlagComputer;    // 1: IBM, 2: Transputer, 3: DEC/UNIX, 4: Mac,
                                // 5: VAX/VMS, 6: GNU/Linux

        RWORD fThetaMin, fThetaMax; // degrees
        RWORD fPhiMin, fPhiMax; // degrees

        RWORD fCerenkovBunch;
        RWORD fCerenkovNumberX, fCerenkovNumberY;
        RWORD fCerenkovGridX, fCerenkovGridY; // cm
        RWORD fCerenkovDetectorX, fCerenkovDetectorY; // cm
        RWORD fCerenkovOutputFlag;

        RWORD fArrayRotation;
        RWORD fFlagExtraMuonInformation;

        RWORD fMultipleScatteringStep;
        RWORD fCerenkovBandwidthMin, fCerenkovBandwidthMax; // nm
        RWORD fUsersOfEvent;
        RWORD fCoreX[20], fCoreY[20]; // cm

        RWORD fFlagSIBYLL, fFlagSIBYLLCross;
        RWORD fFlagQGSJET, fFlagQGSJETCross;
        RWORD fFlagDPMJET, fFlagDPMJETCross;
        RWORD fFlagVENUSCross;
        RWORD fFlagMuonMultiple; // 0: Gauss, 1: Moilere
        RWORD fNKGRadialRange;  // cm
        RWORD fEFractionThinningH; // Energy fraction of thinning
                                   // level
        // These are in the CORSIKA manual but not in Lukas's original code
        RWORD fEFractionThinningEM; // Energy fraction of thinning level EM
        RWORD fWMaxHadronic, fWMaxEM;
        RWORD fRMaxThinning;
        RWORD fInnerAngle, fOuterAngle;

        // Padding adjusted according to additions described above
        RWORD fPad[119 + kPaddingForThinning];
      }; // EventHeader

      /*!
        \struct RunTrailer
        \brief run trailer struct for Corsika files
      */
      struct RunTrailer {
        BlockID fID;
        RWORD fRunNumber;
        RWORD fEventsProcessed;

        RWORD fPad[270 + kPaddingForThinning];
      }; // RunTrailer

      /*!
        \struct EventTrailer
        \brief event trailer struct for Corsika files
      */
      struct EventTrailer {
        BlockID fID;
        RWORD fEventNumber;

        RWORD fPhotons;
        RWORD fElectrons;
        RWORD fHadrons;
        RWORD fMuons;
        RWORD fParticles;

        // NKG output
        RWORD fLateral1X[21], fLateral1Y[21]; // cm^-2
        RWORD fLateral1XY[21], fLateral1YX[21]; // cm^-2

        RWORD fLateral2X[21], fLateral2Y[21]; // cm^-2
        RWORD fLateral2XY[21], fLateral2YX[21]; // cm^-2

        RWORD fElectronNumber[10]; // in steps of 100 g cm^-2
        RWORD fAge[10];         // in steps of 100 g cm^-2
        RWORD fDistances[10];   // cm
        RWORD fLocalAge1[10];

        RWORD fLevelHeightMass[10]; // g cm^-2
        RWORD fLevelHeightDistance[10]; // cm
        RWORD fDistanceBinsAge[10]; // cm
        RWORD fLocalAge2[10];

        // Longitudinal distribution
        RWORD fLongitudinalPar[6];
        RWORD fChi2;
        // Added according to the CORSIKA manual
        RWORD fWeightedPhotons;
        RWORD fWeightedElectrons;
        RWORD fWeightedHadrons;
        RWORD fWeightedMuons;

        RWORD fPad[7 + kPaddingForThinning];
      }; // EventTrailer

      /*!
        \struct ParticleData
        \brief struct with particle data
      */
      struct ParticleData {
        bool IsParticle() const
        { return 0 < fDescription && fDescription < 100000; }
        bool IsNucleus() const
        { return 100000 <= fDescription && fDescription < 9900000; }
        bool IsCherenkov() const
        { return 9900000 <= fDescription; }

        RWORD fDescription;     // Particle ID
        RWORD fPx, fPy, fPz;    // GeV
        RWORD fX, fY;           // cm
        RWORD fTorZ;            // ns or cm
        RWORD fWeight;          // absent if CORSIKA compiled w/o thinning
      }; // ParticleData

      /*!
        \struct ParticleBlock
        \brief block of partile data
      */
      struct ParticleBlock {
        ParticleData fParticle[kParticlesInBlock];
      };

      /*!
        \struct CherenkovData
        \brief struct with Cherenkov data
      */
      struct CherenkovData {
        RWORD fPhotonsInBunch;
        RWORD fX, fY;           // cm
        RWORD fU, fV;           // cos to X and Y axis
        RWORD fT;                       // ns
        RWORD fProductionHeight;        // cm
        RWORD fWeight;          // absent if CORSIKA compiled w/o thinning
      }; // CherenkovData

      /*!
        \struct CherenkovBlock
        \brief block of Cherenkov data
      */
      struct CherenkovBlock {
        CherenkovData fParticle[kParticlesInBlock];
      };

      /*!
        \union SubBlock
        \brief union of blocks
      */
      union SubBlock {
        RunHeader fRunHeader;
        RunTrailer fRunTrailer;
        EventHeader fEventHeader;
        EventTrailer fEventTrailer;
        ParticleBlock fParticleBlock;
        CherenkovBlock fCherenkovBlock;
      }; // SubBlock

      bool IsRunHeader() const { return fSubBlock.fRunHeader.fID.Is("RUNH"); }
      bool IsRunTrailer() const { return fSubBlock.fRunTrailer.fID.Is("RUNE"); }
      bool IsEventHeader() const { return fSubBlock.fEventHeader.fID.Is("EVTH"); }
      bool IsEventTrailer() const { return fSubBlock.fEventTrailer.fID.Is("EVTE"); }
      bool IsControl() const
      { return IsRunHeader() || IsRunTrailer() || IsEventHeader() || IsEventTrailer(); }

      const RunHeader& AsRunHeader() const
      { return fSubBlock.fRunHeader; }
      const RunTrailer& AsRunTrailer() const
      { return fSubBlock.fRunTrailer; }
      const EventHeader& AsEventHeader() const
      { return fSubBlock.fEventHeader; }
      const EventTrailer& AsEventTrailer() const
      { return fSubBlock.fEventTrailer; }
      const ParticleBlock& AsParticleBlock() const
      { return fSubBlock.fParticleBlock; }
      const CherenkovBlock& AsCherenkovBlock() const
      { return fSubBlock.fCherenkovBlock; }

    private:
      SubBlock fSubBlock;

    }; // Block




    class BlockUnthinned {
    public:
      typedef float RWORD;      // 4 byte real
      //typedef int   IWORD;    // 4 byte int

      static const int kMaxObservationLevels = 10;
      static const int kParticlesInBlock  = 39;

      /// Padding for thinning (in RWORDS): 39 (thinning) or 0 (no thinning)
      static const int kPaddingForThinning = 0;

      /*!
        \class BlockID
        \brief Sub-block used in CORSIKA files
      */
      class BlockID {
      public:
        /// Length of sub-block identifier
        static const size_t kLength = 4;

//         /// set from c-string (for testing)
//         void SetID(const char* const theID);

        /// Compare ID's
        bool Is(const char* const theID) const;

      private:
        char fID[kLength];      // hold strings like RUNH etc to
                                // identify the block type
      }; // BlockID

      /*!
        \struct RunHeader
        \brief run header struct for Corsika files
      */
      struct RunHeader {
        BlockID fID;
        RWORD fRunNumber;
        RWORD fDateStart;
        RWORD fVersion;

        RWORD fObservationLevels;
        RWORD fObservationHeight[kMaxObservationLevels]; // in cm

        RWORD fSpectralSlope;
        RWORD fEMin, fEMax;     // in GeV

        RWORD fFlagEGS4;
        RWORD fFlagNKG;

        RWORD fCutoffHadrons;   // in GeV
        RWORD fCutoffMuons;     // in GeV
        RWORD fCutoffElectrons; // in GeV
        RWORD fCutoffPhotons;   // in GeV

        RWORD fConstC[50];
        RWORD fConstCC[20];
        RWORD fConstCKA[40];
        RWORD fConstCETA[5];
        RWORD fConstCSTRBA[11];
        RWORD fConstUNUSED[4];
        RWORD fConstCAN[50];
        RWORD fConstCANN[50];
        RWORD fConstAATM[5];
        RWORD fConstBATM[5];
        RWORD fConstCATM[5];
        RWORD fConstNFLAIN;
        RWORD fConstNFLDIF;
        RWORD fConstNFLPI;
        RWORD fConstNFLCHE;

        RWORD fPad[kPaddingForThinning];
      }; // RunHeader

      /*!
        \struct EventHeader
        \brief event header struct for Corsika files
      */
      struct EventHeader {
        BlockID fID;
        RWORD fEventNumber;
        RWORD fParticleId;
        RWORD fEnergy;
        RWORD fStartingAltitude;        // g cm^-2
        RWORD fFirstTarget;     //
        RWORD fZFirst;          // cm
        RWORD fPx, fPy, fPz;    // GeV
        RWORD fTheta;           // zenith in radians
        RWORD fPhi;             // azimuth in radians

        RWORD fRandomSequences;
        struct {
          RWORD fSeed;
          RWORD fInitialCallsMod;
          RWORD fInitialCallsDiv;
        } fSeeds[10];

        RWORD fRunNumber;
        RWORD fDateStart;
        RWORD fVersion;

        RWORD fObservationLevels;
        RWORD fObservationHeight[kMaxObservationLevels]; // in cm

        RWORD fSpectralSlope;   //
        RWORD fEMin, fEMax;     // in GeV

        RWORD fCutoffHadrons;   // in GeV
        RWORD fCutoffMuons;     // in GeV
        RWORD fCutoffElectrons; // in GeV
        RWORD fCutoffPhotons;   // in GeV

        RWORD fNFLAIN;
        RWORD fNFLDIF;
        RWORD fNFLPI0;
        RWORD fNFLPIF;
        RWORD fNFLCHE;
        RWORD fNFRAGM;

        RWORD fBx, fBz;         // magnetic field in mu T
        RWORD fFlagEGS4;
        RWORD fFlagNKG;

        RWORD fFlagGeisha;
        RWORD fFlagVenus;
        RWORD fFlagCerenkov;
        RWORD fFlagNeutrino;
        RWORD fFlagCurved;
        RWORD fFlagComputer;    // 1: IBM, 2: Transputer, 3: DEC/UNIX, 4: Mac,
                                // 5: VAX/VMS, 6: GNU/Linux

        RWORD fThetaMin, fThetaMax; // degrees
        RWORD fPhiMin, fPhiMax; // degrees

        RWORD fCerenkovBunch;
        RWORD fCerenkovNumberX, fCerenkovNumberY;
        RWORD fCerenkovGridX, fCerenkovGridY; // cm
        RWORD fCerenkovDetectorX, fCerenkovDetectorY; // cm
        RWORD fCerenkovOutputFlag;

        RWORD fArrayRotation;
        RWORD fFlagExtraMuonInformation;

        RWORD fMultipleScatteringStep;
        RWORD fCerenkovBandwidthMin, fCerenkovBandwidthMax; // nm
        RWORD fUsersOfEvent;
        RWORD fCoreX[20], fCoreY[20]; // cm

        RWORD fFlagSIBYLL, fFlagSIBYLLCross;
        RWORD fFlagQGSJET, fFlagQGSJETCross;
        RWORD fFlagDPMJET, fFlagDPMJETCross;
        RWORD fFlagVENUSCross;
        RWORD fFlagMuonMultiple; // 0: Gauss, 1: Moilere
        RWORD fNKGRadialRange;  // cm
        RWORD fEFractionThinningH; // Energy fraction of thinning
                                   // level
        // These are in the CORSIKA manual but not in Lukas's original code
        RWORD fEFractionThinningEM; // Energy fraction of thinning level EM
        RWORD fWMaxHadronic, fWMaxEM;
        RWORD fRMaxThinning;
        RWORD fInnerAngle, fOuterAngle;

        // Padding adjusted according to additions described above
        RWORD fPad[119 + kPaddingForThinning];
      }; // EventHeader

      /*!
        \struct RunTrailer
        \brief run trailer struct for Corsika files
      */
      struct RunTrailer {
        BlockID fID;
        RWORD fRunNumber;
        RWORD fEventsProcessed;

        RWORD fPad[270 + kPaddingForThinning];
      }; // RunTrailer

      /*!
        \struct EventTrailer
        \brief event trailer struct for Corsika files
      */
      struct EventTrailer {
        BlockID fID;
        RWORD fEventNumber;

        RWORD fPhotons;
        RWORD fElectrons;
        RWORD fHadrons;
        RWORD fMuons;
        RWORD fParticles;

        // NKG output
        RWORD fLateral1X[21], fLateral1Y[21]; // cm^-2
        RWORD fLateral1XY[21], fLateral1YX[21]; // cm^-2

        RWORD fLateral2X[21], fLateral2Y[21]; // cm^-2
        RWORD fLateral2XY[21], fLateral2YX[21]; // cm^-2

        RWORD fElectronNumber[10]; // in steps of 100 g cm^-2
        RWORD fAge[10];         // in steps of 100 g cm^-2
        RWORD fDistances[10];   // cm
        RWORD fLocalAge1[10];

        RWORD fLevelHeightMass[10]; // g cm^-2
        RWORD fLevelHeightDistance[10]; // cm
        RWORD fDistanceBinsAge[10]; // cm
        RWORD fLocalAge2[10];

        // Longitudinal distribution
        RWORD fLongitudinalPar[6];
        RWORD fChi2;
        // Added according to the CORSIKA manual
        RWORD fWeightedPhotons;
        RWORD fWeightedElectrons;
        RWORD fWeightedHadrons;
        RWORD fWeightedMuons;

        RWORD fPad[7 + kPaddingForThinning];
      }; // EventTrailer

      /*!
        \struct ParticleData
        \brief struct with particle data
      */
      struct ParticleData {
        bool IsParticle() const
        { return 0 < fDescription && fDescription < 100000; }
        bool IsNucleus() const
        { return 100000 <= fDescription && fDescription < 9900000; }
        bool IsCherenkov() const
        { return 9900000 <= fDescription; }

        RWORD fDescription;     // Particle ID
        RWORD fPx, fPy, fPz;    // GeV
        RWORD fX, fY;           // cm
        RWORD fTorZ;            // ns or cm

      }; // ParticleData

      /*!
        \struct ParticleBlock
        \brief block of partile data
      */
      struct ParticleBlock {
        ParticleData fParticle[kParticlesInBlock];
      };

      /*!
        \struct CherenkovData
        \brief struct with Cherenkov data
      */
      struct CherenkovData {
        RWORD fPhotonsInBunch;
        RWORD fX, fY;           // cm
        RWORD fU, fV;           // cos to X and Y axis
        RWORD fT;                       // ns
        RWORD fProductionHeight;        // cm
      }; // CherenkovData

      /*!
        \struct CherenkovBlock
        \brief block of Cherenkov data
      */
      struct CherenkovBlock {
        CherenkovData fParticle[kParticlesInBlock];
      };

      /*!
        \union SubBlock
        \brief union of blocks
      */
      union SubBlock {
        RunHeader fRunHeader;
        RunTrailer fRunTrailer;
        EventHeader fEventHeader;
        EventTrailer fEventTrailer;
        ParticleBlock fParticleBlock;
        CherenkovBlock fCherenkovBlock;
      }; // SubBlock

      bool IsRunHeader() const { return fSubBlock.fRunHeader.fID.Is("RUNH"); }
      bool IsRunTrailer() const { return fSubBlock.fRunTrailer.fID.Is("RUNE"); }
      bool IsEventHeader() const { return fSubBlock.fEventHeader.fID.Is("EVTH"); }
      bool IsEventTrailer() const { return fSubBlock.fEventTrailer.fID.Is("EVTE"); }
      bool IsControl() const
      { return IsRunHeader() || IsRunTrailer() || IsEventHeader() || IsEventTrailer(); }

      const RunHeader& AsRunHeader() const
      { return fSubBlock.fRunHeader; }
      const RunTrailer& AsRunTrailer() const
      { return fSubBlock.fRunTrailer; }
      const EventHeader& AsEventHeader() const
      { return fSubBlock.fEventHeader; }
      const EventTrailer& AsEventTrailer() const
      { return fSubBlock.fEventTrailer; }
      const ParticleBlock& AsParticleBlock() const
      { return fSubBlock.fParticleBlock; }
      const CherenkovBlock& AsCherenkovBlock() const
      { return fSubBlock.fCherenkovBlock; }

    private:
      SubBlock fSubBlock;

    }; // Block



  }; // namespace Corsika

} // io


#endif // _io_Corsika_Block_h_

// Configure (x)emacs for this file ...
// Local Variables:
// mode: c++
// compile-command: "make -C .. -k"
// End:
