#ifndef _t2_triggerntuple_
#define _t2_triggerntuple_

#include <Rtypes.h>
#include <vector>
#include <stdexcept>
#include <algorithm>


namespace t2 {

  struct TriggerNTuple {
    unsigned int fGPSSecond = 0;
    unsigned int fMicroSecond = 0;    //> defined as us of first trigger
    unsigned short fId = 0;

    std::vector<int> fDeltaT1;        //> time difference to first trigger: t21, t31, t41, ...
    std::vector<short> fTriggerTypes; //> t1, t2, t3

    TriggerNTuple() = default;

    // note that gps second has to be set elsewhere
    template<class T>
    TriggerNTuple(const std::vector<T>& triggers, const uint gps = 0)
      : fGPSSecond(gps)
    {
      if (!std::is_sorted(triggers.begin(), triggers.end()))
        throw std::runtime_error("TriggerTuple(): vector not sorted!");

      fId = triggers.front().fId;
      fMicroSecond = triggers.front().fTime;

      const auto& t1 = triggers.front();
      for (int i = 1; i < triggers.size(); ++i)
        fDeltaT1.push_back(triggers[i].fTime - t1.fTime);

      for (const auto& t : triggers)
        fTriggerTypes.push_back(t.fTriggers);
    }

    ClassDefNV(TriggerNTuple, 1)
  };

}

#endif