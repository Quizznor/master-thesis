#include <interface/CandidateJoinedWithCDAS.h>

ClassImp(t2::CandidateJoinedWithCDAS)

namespace t2 {

  bool 
  CandidateJoinedWithCDAS::HasLostStationData() 
    const
  {
    for (const auto& t2 : fT3.fT2s) {
      if (!t2)  //no infill!
        continue;
      if (!t2.fTime)
        return true;
    }
    return false;
  }
};