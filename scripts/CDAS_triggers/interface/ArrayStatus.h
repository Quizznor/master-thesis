#ifndef _ArrayStatus_h_
#define _ArrayStatus_h_

#include <Rtypes.h>
#include <vector>
#include <exception>
#include <iostream>
#include <io/RootOutFile.h>


namespace t2 {

  struct DeltaT {
    uint fGPSSecond = 0;
    uint fMeanMicroSecond = 0;
    uint fId = 0;

    double fDeltaT = 0;
    ushort fTriggersOld = 0;
    ushort fTriggersNew = 0;

    bool fClosestTrigger = false;

    template<class t>
    void
    FillData(const t& n, const t& old)
    {
      if (n.fId != old.fId)
        throw std::logic_error("mismatching ids!");

      fMeanMicroSecond = (n.fTime + old.fTime)/ (2 * kMicroSecond);
      fId = n.fId;
      fDeltaT = (n.fTime - old.fTime)/kMicroSecond;
      fTriggersOld = old.fTriggers;
      fTriggersNew = n.fTriggers;

      if (fDeltaT < 0)
        std::cerr << "warning negative delta: "
                  << fDeltaT << " us "
                  << n.fTime/kMicroSecond << " new us "
                  << old.fTime/kMicroSecond << " old us"
                  << std::endl;
    }

    ClassDefNV(DeltaT, 3);
  };


  struct ArrayStatus {
    uint fGPSSecond = 0;

    uint fnT2 = 0;
    ushort fnWide = 0;
    ushort fnHighGainSaturated = 0;
    uint fnRejected = 0;
    uint fnOutOfGrid = 0;

    uint fId[2000] = { 0 };
    uint fToTId[2000] = { 0 };

    std::vector<ushort> fSilentIds;
    uint fnSendingData = 0;

    uint fnTriplets = 0;
    uint fnCluster = 0;
    uint fnEvents = 0;
    uint fnT3 = 0;

    bool fAbortedReconstruction = false;

    std::vector<T2Data> fLastTrigger; //!
    std::vector<T2Data> fLastTriggerTypeResolved[16]; //!


    ArrayStatus()
    {
      fLastTrigger.resize(2000);
      for (int i = 0; i < 16; ++i)
        fLastTriggerTypeResolved[i].resize(2000);
    }

    void
    SetNWithData()
    {
      for (int i = 0; i < 2000; ++i) {
        if (fId[i])
          ++fnSendingData;
      }
    }

    int
    GetNWithData()
      const
    {
      int n = 0;
      for (int i = 0; i < 2000; ++i) {
        if (fId[i])
          ++n;
      }
      return n;
    }

    template<class t>
    void
    CountT2(const t& t2, io::RootOutFile<DeltaT>& output)
    {
      CountT2(t2);
      if (t2.fId > 0 && t2.fId < 2000) {
        if (!fLastTrigger[t2.fId]) {
          fLastTrigger[t2.fId] = t2;
          fLastTriggerTypeResolved[t2.fTriggers][t2.fId] = t2;
          return;
        }

        DeltaT del;
        del.fGPSSecond = fGPSSecond;
        del.FillData(t2, fLastTrigger[t2.fId]);
        del.fClosestTrigger = true;
        output << del;

        del.fClosestTrigger = false;

        for (int i = 0; i < 16; ++i) {
          if (fLastTriggerTypeResolved[i][t2.fId]) {
            del.FillData(t2, fLastTriggerTypeResolved[i][t2.fId]);
            output << del;
          }
        }

        fLastTrigger[t2.fId] = t2;
        fLastTriggerTypeResolved[t2.fTriggers][t2.fId] = t2;
      }
    }

    template<class t>
    void
    CountT2(const t& t2, std::vector<DeltaT>& output)
    {
      CountT2(t2);

      if (t2.fId > 0 && t2.fId < 2000) {
        if (!fLastTrigger[t2.fId]) {
          fLastTrigger[t2.fId] = t2;
          fLastTriggerTypeResolved[t2.fTriggers][t2.fId] = t2;
          return;
        }

        DeltaT del;
        del.fGPSSecond = fGPSSecond;
        del.FillData(t2, fLastTrigger[t2.fId]);
        del.fClosestTrigger = true;
        output.push_back(del);

        del.fClosestTrigger = false;

        for (int i = 0; i < 16; ++i) {
          if (fLastTriggerTypeResolved[i][t2.fId]) {
            del.FillData(t2, fLastTriggerTypeResolved[i][t2.fId]);
            output.push_back(del);
          }
        }

        fLastTrigger[t2.fId] = t2;
        fLastTriggerTypeResolved[t2.fTriggers][t2.fId] = t2;
      }
    }

    template<class t>
    void
    CountT2(const t& t2)
    {
      if (t2.fId > 2000)
        return;
      ++fId[t2.fId];
      ++fnT2;

      if (t2.IsToT())
        ++fToTId[t2.fId];

      if (t2.IsWide())
        ++fnWide;

      if (t2.IsSaturated())
        ++fnHighGainSaturated;
    }

    void
    AdvanceSecond()
    {
      for (auto& t : fLastTrigger) {
        t.fTime -= kOneSecond*kMicroSecond;
        if (t.fId
          && (t.fTime > 0 || t.fTime < -5*kOneSecond*kMicroSecond)) {
          std::cerr << "warning resetting trigger: "
                    << t.fTime << " us. "
                    << t.fId << " id"
                    << std::endl;
          t = T2Data();
        }
      }

      for (int i = 0; i < 16; ++i) {
        for (auto& t : fLastTriggerTypeResolved[i]) {
          t.fTime -= kOneSecond*kMicroSecond;
          if (t.fId
          && (t.fTime > 0 || t.fTime < -5*kOneSecond*kMicroSecond)) {

            t = T2Data();
          }
        }
      }

      ++fGPSSecond;

      fnT2 = 0;
      fnRejected = 0;
      fnOutOfGrid = 0;

      for (auto& n : fId)
        n = 0;

      for (auto& n : fToTId)
        n = 0;

      fSilentIds.clear();
      fnSendingData = 0;

      fnTriplets = 0;
      fnCluster = 0;
      fnEvents = 0;
      fnT3 = 0;

      fAbortedReconstruction = false;
    }

    ClassDefNV(ArrayStatus, 6);
  };

  inline
  std::ostream&
  operator<<(std::ostream& os, const ArrayStatus& s)
  {
    os << s.fGPSSecond << " "
       << s.fnT2 << " "
       << s.fnRejected << " "
       << s.fnSendingData << " "
       << s.fnTriplets << " "
       << s.fnT3 << " "
       << s.fnCluster << " "
       << s.fnEvents;
    return os;
  }
};
#endif