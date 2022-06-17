#ifndef _Events_
#define _Events_

#include <Rtypes.h>
#include <cmath>
#include <interface/T2.h>
#include <interface/RecoT2.h>
#include <interface/EventCandidate.h>
#include <TVector3.h>
#include <TH2D.h>

typedef unsigned short ushort;

struct Positions {
  TVector3 fPosition[2000];
  TVector3& operator[](int i) { return fPosition[i]; }
  const TVector3& operator[](int i) const { return fPosition[i]; }
};

//Event Classes for MC and Analysis output
struct Event {
  uint fGPSSecond = 0;
  uint fMicroSecond = 0;

  double fu = 0;
  double fv = 0;

  double GetTheta() const { return acos(sqrt(1 - fu*fu - fv*fv)); }
  double GetPhi() const { return atan(fv/fu); }

  ClassDefNV(Event, 1);
};

//not inhereting from Event, as MC works in theta/phi while analysis in u,v space
struct MCEvent {
  uint fGPSSecond = 0;
  uint fMicroSecond = 0;

  double fTheta = 0;
  double fPhi = 0;  //or height of lightning-explosion

  std::vector<ushort> fIds;
  std::vector<std::pair<int, int>> fPositions;
  std::vector<int> fTimes;

  short fType = 0; // 0: compact. 1: extended. 
                   // 2: "spallation". 3: lightning (spherical)
                   // 4: horizontal event
  bool fCompact = false;

  bool operator<(const MCEvent& m) const
  {
    return double(fGPSSecond) - double(m.fGPSSecond) 
            + 1e-6*(double(fMicroSecond) - double(m.fMicroSecond)) < 0; 
  }

  int
  GetKey()
    const
  {
    return fType*100 + fIds.size();
  }

  ClassDefNV(MCEvent, 3);
};

//for output of event candidates
// includes additional information, to refine the reconstruction 'offline'
struct T2EventCandidate : public Event {
  std::vector<ReconstructedT2> fReconstructedTriplets;
  std::vector<T2> fT2s;
  //std::vector<ushort> fSignalIds;

  int fSignalClass = 0;   //To be defined

  int fMaxAdditionalMatches = 0 ;
  int fTestStatistic = 0; //current approach: number of triplets with 3 add.Matches + 2*n(4) + 3*n(5) + ...

  int fnInCluster = 0;
  int fnCluster = 0;
  int fnMaxPerCluster = 0;

  double fAvgDistance = 0;

  void AddReconstructedTriplet(const ReconstructedT2& reco);
  void AddReconstructedTripletMeterBased(const ReconstructedT2& r);

  void Reconstruct(); //direction and time
  bool IsCompatibleWithTimeRange(const ReconstructedT2& r, int windowsize = 500, double conversionFactor = 1) const;
  double GetAvgDistance();
  int GetNCompact(double threshold = 6.) const;
  ushort GetNIdsWithMultipleT2s() const;

  TH2D HughTransformLineSearch(const Positions&) const;

  /*void Classify();
  int Classify() const;*/

  bool IsT3() const;
  bool ContainsT3(const Positions&) const;  //differentiate between triplets with one station of a T3-set and
                            // and sample with T3

  int Classify();
  void Clear();

  ClassDefNV(T2EventCandidate, 3);
};

struct JoinedEvent : public T2EventCandidate {
  uint fGPSSecondMC = 0;
  uint fMicroSecondMC = 0;

  double fThetaMC = 0;
  double fPhiMC = 0;
  std::vector<ushort> fMCIds;

  bool fCompact = false;

  JoinedEvent() {}
  ~JoinedEvent() {}
  
  JoinedEvent(const T2EventCandidate& cand, const MCEvent& mc)
  : 
    T2EventCandidate(cand),
    fGPSSecondMC(mc.fGPSSecond),
    fMicroSecondMC(mc.fMicroSecond),
    fThetaMC(mc.fTheta),
    fPhiMC(mc.fPhi),
    fMCIds(mc.fIds),
    fCompact(mc.fCompact)
  {
  }

  ClassDefNV(JoinedEvent, 1);
};

struct OfflineMatchedEvent : public T2EventCandidate {
  uint fGPSSecondOffline = 0;
  uint fMicroSecondOffline = 0;

  ushort fnStationsOffline = 0;

  OfflineMatchedEvent() {}
  OfflineMatchedEvent(const T2EventCandidate& cand) :
  T2EventCandidate(cand)
  {}
  ~OfflineMatchedEvent() {}

  ClassDefNV(OfflineMatchedEvent, 1);
};

struct ScalerMatchedEvent : public T2EventCandidate, EventCandidate {
  ScalerMatchedEvent() {}
  ~ScalerMatchedEvent() {}
  ScalerMatchedEvent(const T2EventCandidate& c, const EventCandidate& cScaler)
  : T2EventCandidate(c),
    EventCandidate(cScaler)
    {}

  ClassDefNV(ScalerMatchedEvent, 1);
};


#endif
