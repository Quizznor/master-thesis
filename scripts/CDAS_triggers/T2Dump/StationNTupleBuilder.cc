#include <T2Dump/StationNTupleBuilder.h>
#include <stdexcept>
#include <cmath>


namespace t2 {


  t2::TriggerNTuple
  StationNTupleBuilder::operator()(const T2Data& t)
  {
    if (!fPreviousTrigger.size()) {
      fPreviousTrigger.push_back(t);
      return t2::TriggerNTuple();
    }

    const auto dt1 = t.fTime - fPreviousTrigger.front().fTime;
    if (fDeltaT1Max > 0) {
      if (dt1 < fDeltaT1Max) {
        fPreviousTrigger.push_back(t);
        return t2::TriggerNTuple();
      } else {
        t2::TriggerNTuple tmp(fPreviousTrigger);
        fPreviousTrigger.clear();
        fPreviousTrigger.push_back(t);
        return tmp;
      }
    } else if (fMaxHyperRadius > 0) {
      fCurrentTSqr += std::pow(dt1, 2);
      const auto hyperRad = std::sqrt(fCurrentTSqr);
      if (hyperRad < fMaxHyperRadius) {
        fPreviousTrigger.push_back(t);
        return t2::TriggerNTuple();
      } else {
        t2::TriggerNTuple tmp(fPreviousTrigger);
        fPreviousTrigger.clear();
        fPreviousTrigger.push_back(t);
        fCurrentTSqr = 0;
        return tmp;
      }
    } else {
      std::runtime_error("no valid condition for StationNTupleBuilder!");
    }
  }
}