#ifndef _t2_stationntuplebuilder_h_
#define _t2_stationntuplebuilder_h_

#include <interface/TriggerNTuple.h>
#include <t2/T2Data.h>
#include <vector>
#include <limits>


namespace t2 {


  class StationNTupleBuilder {
    std::vector<t2::T2Data> fPreviousTrigger;
    double fCurrentTSqr = 0;   //> pre compute the hyper-radius to save some time

  public:
    StationNTupleBuilder() = default;

    void
    AdvanceSecond()
    {
      for (auto& t : fPreviousTrigger) {
        t -= 1e6;
        // underflow check
        if (t > 0)
          fPreviousTrigger.clear();
      }
    }

    t2::TriggerNTuple operator()(const T2Data& t);

    double fDeltaT1Max = 2000/*us*/;
    double fMaxHyperRadius = -1/*us*/;  // t_r = sqrt(Sum_i t_{i1}^2), -1 -> don't use
  };

}

#endif