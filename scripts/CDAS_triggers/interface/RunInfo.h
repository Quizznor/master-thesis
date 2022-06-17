#ifndef _RunInfo_h_
#define _RunInfo_h_

#include <Rtypes.h>
#include <string>
#include <vector>
#include <T2Dump/GraphSearch.h>

namespace t2 {

  struct RunInfo {
    //DB-Scan (directional cluster)
    double fEpsilon = 0;
    int fmPts = 0;
    double fTolerance = 0;
    double fdbCut = 0;

    //Compact searches (spatial cluster):
    int fMinT2PerGraph = 0;
    int fMaxCrown = 0;
    bool fOnlyCoherent = 0;
    double fMaxDeltaTLightning = 0;

    int fSeed = 0;
    bool fBackground = false;

    std::string fOutputBase = "";
    std::vector<std::string> fInputFiles;
    std::string fStationFile = "";

    uint fnReconstructedTriplets = 0;
    uint fnEvents = 0;
    uint fnAnalysedSeconds = 0;
    uint fnAnalysedSecondsWithData = 0; //more than 600 stations
    uint fnMaxT3InOneSecond = 0;
    uint fnAbortedReconstruction = 0;

    void
    FillGraphSerachParameters(const GraphSearch& gs)
    {
      fMinT2PerGraph = gs.fMinimalNumberPerGraph;
      fMaxCrown = gs.fMaxCrown;
      fOnlyCoherent = gs.fOnlyCoherent;
      fMaxDeltaTLightning = gs.fMaxDeltaTLightning;
    }

    bool
    operator==(const RunInfo& b)
      const
    {
      return fEpsilon == b.fEpsilon &&
             fmPts == b.fmPts &&
             fTolerance == b.fTolerance &&
             fdbCut == b.fdbCut;
    }

    bool
    operator!=(const RunInfo& b)
      const
    {
      return !(*this == b);
    }

    bool
    IsLightning(const int nT3Threshold = 5)
      const
    {
      return    int(fnMaxT3InOneSecond) > nT3Threshold
             || fnAbortedReconstruction;
    }

    ClassDefNV(RunInfo, 3);
  };
};
#endif