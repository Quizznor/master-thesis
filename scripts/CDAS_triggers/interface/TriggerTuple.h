#ifndef _t2_trigger_tuple_
#define _t2_trigger_tuple_

#include <Rtypes.h>
#include <vector>
#include <stdexcept>
#include <algorithm>


namespace t2 {

  // t1 t2 t3 -> t3 - t1 ==> fDeltaT31
  struct TriggerTuple {
    unsigned int fGPSSecond = 0;
    unsigned int fMicroSecond = 0;    //> defined as us of last trigger
    unsigned short fId = 0;

    int fDeltaT31 = 0;
    int fDeltaT21 = 0;

    std::vector<short> fTriggerTypes; //> t1, t2, t3

    TriggerTuple() = default;
    TriggerTuple(const TriggerTuple&) = default;

    template<class T>
    TriggerTuple(const std::vector<T>& triggers)
    {
      if (triggers.size() != 3)
        throw std::runtime_error("TriggerTuple(): vector size != 3");

      if (!std::is_sorted(triggers.begin(), triggers.end()))
        throw std::runtime_error("TriggerTuple(): vector not sorted!");

      fId = triggers.front().fId;
      fMicroSecond = triggers.back().fTime;
      fDeltaT31 = triggers.back() - triggers.front();
      fDeltaT21 = triggers[1] - triggers[0];


      for (const auto& t : triggers)
        fTriggerTypes.push_back(t.fTriggers);
    }

    ClassDefNV(TriggerTuple, 2)
  };

}

#endif