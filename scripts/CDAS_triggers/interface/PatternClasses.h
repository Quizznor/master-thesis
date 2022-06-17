#ifndef _PatternClasses_h_
#define _PatternClasses_h_

#include <map>
#include <iostream>
#include <interface/RunInfo.h>
#include <interface/Graph.h>
#include <exception>
#include <Rtypes.h>
#include <string>
#include <T2Dump/Utl.h>


namespace t2 {

  const bool kIncludeNonWorkingStations = true;

  struct PatternClasses {
    std::map<std::string, uint> fMapCounts;
    std::map<std::string, uint> fMapMinCrown;
    std::map<std::string, uint> fMapMaxCrown;
    std::map<std::string, uint> fMapNTot;
    std::map<std::string, bool> fMapT3;

    uint fnCandidates = 0;
    uint fnRejected = 0;

    RunInfo fRunInfo;

    void
    AddCandidate(const Graph& g)
    {
      /*if (g.fDeadNeighbours.size()) {
        ++fnRejected;
        return;
      }*/

      ++fnCandidates;
      const auto t3 = 0; //g.ContainsT3();
      std::string t3Name = "";
      switch (t3) {
        case 0:
          t3Name = "";
        break;
        case 1:
          t3Name = "4T2";
        break;
        case 2:
          t3Name = "3ToT";
        break;
      }

      const auto key = t3Name.empty() ?
        g.GetMinKey(kIncludeNonWorkingStations) : t3Name;

      const auto it = fMapCounts.find(key);
      if (it != fMapCounts.end()) {
        ++it->second;
        return;
      } else {
        const bool useZeroCrowns = true;
        fMapCounts.insert(std::make_pair(key, 1));
        fMapMaxCrown.insert(std::make_pair(key, g.GetMinimalCrownOfPattern()));
        fMapMinCrown.insert(std::make_pair(key, g.GetMinimalCrown(useZeroCrowns)));
        fMapNTot.insert(std::make_pair(key,
                          round(g.GetToTFraction()*g.fNodesSignal.size())));
        if (g.ContainsT3(false, true)) //ignore timing
          fMapT3.insert(std::make_pair(key, true));
        else
          fMapT3.insert(std::make_pair(key, false));
      }

      return;
    }

    bool
    IsNormal(const double minOnTimeFraction = 0.75)
      const
    {
      if (!fRunInfo.fBackground) {
        std::cerr << "#warning: Asking about quality flag in non-background output!"
                  << std::endl;
        return true;
      }

      if (fRunInfo.fnAbortedReconstruction)
        return false;

      if (fRunInfo.fnAnalysedSecondsWithData
            < minOnTimeFraction*fRunInfo.fnAnalysedSeconds)
        return false;

      if (fRunInfo.fnMaxT3InOneSecond > 4)
        return false;

      return true;
    }

    ClassDefNV(PatternClasses, 5);
  };

  inline
  std::ostream&
  operator<<(std::ostream& os, const PatternClasses& p)
  {
    if (p.fRunInfo.fBackground)
      os << "Bg: ";
    os << "in " << p.fRunInfo.fnAnalysedSeconds
       << " seconds " << p.fMapCounts.size()
       << " different patterns with " << p.fnCandidates
       << " candidates"
       << std::endl;
    for (const auto& entry : p.fMapCounts)
      os << ' ' << entry.first << " ("
         << p.fMapMaxCrown.find(entry.first)->second << " maxCrown, "
         << p.fMapMinCrown.find(entry.first)->second << " minCrown)"
         << ": " << entry.second << std::endl;

    return os;
  }
};
#endif