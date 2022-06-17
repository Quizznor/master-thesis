#ifndef _CandidateClasses_h_
#define _CandidateClasses_h_

#include <map>
#include <iostream>
#include <interface/RunInfo.h>
#include <interface/MergedCandidate.h>
#include <exception>
#include <Rtypes.h>
#include <string>


namespace t2 {

  struct CandidateClasses {
    std::map<std::string, uint> fCounts; 
    uint fnCandidates = 0;

    RunInfo fRunInfo;

    void
    AddCandidate(const MergedCandidate& m)
    {
      ++fnCandidates;
      const auto key = m.GenerateKey();
      auto it = fCounts.find(key);
      if (it == fCounts.end()) {
        fCounts.insert(std::make_pair(key, 1));
      } else {
        ++it->second;
      }
    }

    uint
    GetNT3()
      const
    {
      uint tmp = 0;
      for (const auto& p : fCounts) {
        if (p.first == "3ToT" || p.first == "4T2" || p.first == "T3")
          tmp += p.second;
      }
      return tmp;
    }

    bool
    IsNormal(double minOnTimeFraction = 0.75)
      const
    {
      if (!fRunInfo.fBackground) {
        std::cerr << "warning: Asking about quality flag"
                     "in non-background output!"
                  << std::endl;
        return true;
      }

      if (fRunInfo.fnAbortedReconstruction)
        return false;

      if (fRunInfo.fnAnalysedSecondsWithData 
            < minOnTimeFraction*fRunInfo.fnAnalysedSeconds)
        return false;

      if (GetNT3() > 0.02*fRunInfo.fnAnalysedSecondsWithData)
        return false;

      if (fRunInfo.fnMaxT3InOneSecond > 4)
        return false;

      return true;
    }

    bool IsLightningFile() const
    { return !IsNormal(0); /*ignore on-Time fraction*/ }

    ClassDefNV(CandidateClasses, 2);
  };

  inline
  std::ostream&
  operator<<(std::ostream& os, const CandidateClasses& c)
  {
    if (c.fRunInfo.fBackground)
      os << "Bg: ";
    os << "in " << c.fRunInfo.fnAnalysedSeconds << " seconds" << std::endl;
    for (const auto& p : c.fCounts) 
      os << ' ' << p.first << ' ' << p.second << std::endl;
    
    return os;
  }
};


  
#endif