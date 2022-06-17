#ifndef _binnedT2Counts_h_
#define _binnedT2Counts_h_

#include <Rtypes.h>
#include <vector>
#include <exception>
#include <iostream>
#include <T2Dump/Units.h>


namespace t2 {


  struct BinnedT2Counts {
    uint fGPSSecond = 0;
    uint fStartMicroSecond = 0;
    uint fStopMicroSecond = 0;

    uint fnT2 = 0;
    uint fnToT = 0;

    uint fnSendingData = 0;

    BinnedT2Counts() {}

    BinnedT2Counts(const uint gps)
     : fGPSSecond(gps)
    {
    }

    BinnedT2Counts(const uint gps, const uint start, const uint stop)
     : fGPSSecond(gps), fStartMicroSecond(start), fStopMicroSecond(stop)
    {
    }

    template<class t>
    bool
    CountT2(const t& t2)
    {
      if (   t2.fTime >= fStopMicroSecond * kMicroSecond
          || t2.fTime < fStartMicroSecond * kMicroSecond)
        return false;
      if (t2.fId > 0 && t2.fId < 2000) {
        ++fnT2;

        if (t2.IsToT())
          ++fnToT;
      }
      return true;
    }

    BinnedT2Counts
    GetNextBinObject()
      const
    {
      const int dt = int(fStopMicroSecond) - int(fStartMicroSecond);
      return BinnedT2Counts(fGPSSecond, fStopMicroSecond, fStopMicroSecond + dt);
    }

    ClassDefNV(BinnedT2Counts, 1);
  };

};
#endif