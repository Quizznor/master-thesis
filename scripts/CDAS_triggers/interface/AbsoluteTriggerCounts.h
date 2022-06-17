#ifndef _t2_absolutetriggercounts_h_
#define _t2_absolutetriggercounts_h_

#include <Rtypes.h>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <t2/T2Data.h>


namespace t2 {

  struct AbsoluteTriggerCounts {
    unsigned int fGPSSecondStart = 0;
    unsigned int fGPSSecondEnd = 0;
    std::vector<unsigned long int> fStationCounts;

    std::vector<unsigned long int> fCountsPerTriggerType;

    AbsoluteTriggerCounts() :
      fStationCounts(2000, 0),
      fCountsPerTriggerType(16, 0)
    {
    }

    void
    operator()(const t2::T2Data& t2)
    {
      ++fStationCounts.at(t2.fId);
      ++fCountsPerTriggerType.at(t2.fTriggers);
    }

    ClassDefNV(AbsoluteTriggerCounts, 1)
  };

}

#endif