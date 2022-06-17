#ifndef _t2dump_t2_stationtimedifferencebuilder_h_
#define _t2dump_t2_stationtimedifferencebuilder_h_

#include <interface/TriggerTuple.h>
#include <interface/TriggerNTuple.h>
#include <t2/T2Data.h>
#include <vector>
#include <limits>


namespace t2 {


  template<class T = int>
  class StationTimeDifferenceBuilder {
    std::vector<T> fPreviousUS;
    int fnPrevious = 0;
    int fNDelta = 2;

  public:
    StationTimeDifferenceBuilder()
      : fPreviousUS(2)
    {
    }

    StationTimeDifferenceBuilder(const int n)
      : fPreviousUS(n), fNDelta(n)
    {
    }

    StationTimeDifferenceBuilder(const int n, const int us)
      : fPreviousUS(n), fNDelta(n)
    {
      fPreviousUS[0] = us;
      ++fnPrevious;
    }

    void
    AdvanceSecond()
    {
      for (auto& t : fPreviousUS) {
        t -= 1e6;
        // underflow check
        if (t > 0) {
          t = T();
          fnPrevious = 0;
        }
      }
    }


    // all operators() return 0 on invalid values
    //  e.g. because not enough triggers (requested t31, had only 2 triggers)
    //  are in the buffer
    double
    operator()(const T& t)
    {
      if (fnPrevious < fNDelta) {
        for (int i = fnPrevious; i > 0; --i) {
          fPreviousUS[i] = fPreviousUS[i - 1];
        }
        fPreviousUS[0] = t;
        ++fnPrevious;
        return 0;
      }

      const int dt = t - fPreviousUS[fNDelta - 1];
      for (int i = fNDelta - 1; i > 0; --i) {
        fPreviousUS[i] = fPreviousUS[i - 1];
      }
      fPreviousUS[0] = t;
      ++fnPrevious;

      if (dt < 0)
        return 0;
      return dt;
    }

    double
    operator()(const T& t, TriggerTuple& tuple)
    {
      if (fnPrevious < fNDelta) {
        for (int i = fnPrevious; i > 0; --i) {
          fPreviousUS[i] = fPreviousUS[i - 1];
        }
        fPreviousUS[0] = t;
        ++fnPrevious;
        return 0;
      }

      std::vector<T> tmp;
      for (int i = fNDelta - 1; i >= 0; --i)
        tmp.push_back(fPreviousUS[i]);
      tmp.push_back(t);
      if (tmp.size() < 3) // this is a hack to avoid recoding the interface
        tmp.push_back(t);
      tuple = TriggerTuple(tmp);

      const int dt = t - fPreviousUS[fNDelta - 1];
      for (int i = fNDelta - 1; i > 0; --i) {
        fPreviousUS[i] = fPreviousUS[i - 1];
      }
      fPreviousUS[0] = t;
      ++fnPrevious;

      if (dt < 0)
        return 0;
      return dt;
    }

    double
    operator()(const T& t, TriggerNTuple& tuple)
    {
      if (fnPrevious < fNDelta) {
        for (int i = fnPrevious; i > 0; --i)
          fPreviousUS[i] = fPreviousUS[i - 1];
        fPreviousUS[0] = t;
        ++fnPrevious;
        return 0;
      }

      std::vector<T> tmp;
      for (int i = fNDelta - 1; i >= 0; --i)
        tmp.push_back(fPreviousUS[i]);
      tmp.push_back(t);

      tuple = TriggerNTuple(tmp);

      const int dt = t - fPreviousUS[fNDelta - 1];
      for (int i = fNDelta - 1; i > 0; --i) {
        fPreviousUS[i] = fPreviousUS[i - 1];
      }
      fPreviousUS[0] = t;
      ++fnPrevious;

      if (dt < 0)
        return 0;
      return dt;
    }

    double
    operator()(const int us, TriggerTuple& tuple)
    {
      if (fnPrevious < fNDelta) {
        for (int i = fnPrevious; i > 0; --i) {
          fPreviousUS[i] = fPreviousUS[i - 1];
        }
        fPreviousUS[0] = us;
        ++fnPrevious;
        return 0;
      }

      tuple.fMicroSecond = us;

      const int dt = us - fPreviousUS[fNDelta - 1];
      for (int i = fNDelta - 1; i > 0; --i) {
        fPreviousUS[i] = fPreviousUS[i - 1];
      }
      fPreviousUS[0] = us;
      ++fnPrevious;

      tuple.fDeltaT31 = dt;

      if (dt < 0)
        return 0;
      return dt;
    }

  };

}

#endif