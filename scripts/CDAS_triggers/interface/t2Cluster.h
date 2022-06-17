#ifndef _t2Cluster_h_
#define _t2Cluster_h_

#include <t2/T2Data.h>
#include <T2Dump/Triplet.h>
#include <vector>
#include <algorithm>
#include <T2Dump/Utl.h>
#include <t2/StationInfo.h>
#include <T2Dump/ClusterPoint.h>
#include <T2Dump/GraphNode.h>
#include <utl/Accumulator.h>
#include <T2Dump/Units.h>


namespace t2 {

  struct Cluster {
    uint fGPSSecond = 0;
    double fMicroSecond = 0;

    double fu = 0;
    double fv = 0;

    std::vector<T2Data> fT2s;
    std::vector<GraphNode> fDeadNeighbours;
    std::vector<ClusterPoint> fData;

    bool fUsedInMerging = false; //!

    bool
    operator<(const Cluster& b)
      const 
    {
      const double deltaT = (fGPSSecond - b.fGPSSecond) + (fMicroSecond - b.fMicroSecond)*1e-6; 
      return deltaT < 0;
    }

    bool 
    operator>(const Cluster& b) 
      const 
    {
      const double deltaT = (fGPSSecond - b.fGPSSecond) + (fMicroSecond - b.fMicroSecond)*1e-6; 
      return deltaT > 0;
    }

    void 
    AddTriplet(const rTriplet& r) 
    {
      auto tmp = ClusterPoint(r);
      const double weightOld = fData.size()/double(fData.size() + 1);
      const double weightNew = 1./double(fData.size() + 1);

      fMicroSecond = weightOld*fMicroSecond + tmp.ft0*weightNew;
      fu = weightOld*fu + tmp.fu*weightNew;
      fv = weightOld*fv + tmp.fv*weightNew;

      fData.push_back(tmp);

      for (int i = 0; i < 3; ++i) {
        if (!std::count(fT2s.begin(), fT2s.end(), r.fTrigger[i]))
          fT2s.push_back(r.fTrigger[i]);
      }
    }

    void
    Clear()
    {
      fGPSSecond = 0;
      fMicroSecond = 0;

      fu = 0;
      fv = 0;

      fData.clear();
      fT2s.clear();
    }

    bool
    IsInteresting() 
      const 
    { 
      if (fUsedInMerging)
        return false;
      return fData.size() || fT2s.size(); 
    }

    template <typename T>
    bool 
    ContainsT3(const std::vector<StationInfo<T> >& stationInfos)
      const 
    {
      for (auto it = fT2s.begin(); it != fT2s.end(); ++it) {
        int nNeighboursToT[2] = {0, 0};         //count stations in C1, C2
        int nNeighboursAny[4] = {0, 0, 0, 0};   // ... C1, C2, C3, C4
      
        for (auto it2 = fT2s.begin(); it2 != fT2s.end(); ++it2) {
          if (it2->fId == it->fId)
            continue;

          const auto& p1 = stationInfos[it->fId].fPosition;
          const auto& p2 = stationInfos[it2->fId].fPosition;

          const int crownCDAS = crown(p1.fX, p2.fX, p1.fY, p2.fY);
          const int deltaT = it->fTime/300. - it2->fTime/300.;

          //check compactness and timing conditions
          // hexagons are approximated with circles -> overestimation of compactness
          // time condition is (3 + 5*n) mu s, in CDAS XbAlgo.cc (<= 5*neighbour + dtime)
          // with dtime = 3
          if (abs(deltaT) > 3 + 5*crownCDAS) //directly remove out of time candidates
            continue; 
          
          if (crownCDAS > 4 || crownCDAS < 1)
            continue;

          if (it2->IsToT()) {
            for (int i = crownCDAS - 1; i < 2; ++i)
              ++nNeighboursToT[i];
          }

          for (int i = crownCDAS - 1; i < 4; ++i) 
            ++nNeighboursAny[i];
        } 
        
        //3ToT trigger (using, that the center is triggered by construction)
        if (it->IsToT()
            && nNeighboursToT[0] >= 1 
            && nNeighboursToT[1] >= 2)
          return true;

        //4T2 mode 2C1&3C2&4C4
        if (nNeighboursAny[0] >= 1 
            && nNeighboursAny[1] >= 2 
            && nNeighboursAny[3] >= 3)
          return true;
      }
      return false;
    }

    void SetGPSSecond(uint gps) { fGPSSecond = gps; }

    //returns the smallest distance in time of T to a 
    // point in the cluster
    template<class T>
    int 
    TimeDistance(const T& t) 
      const
    {
      int distance = 1e6;
      for (const auto& point : fData) {
        if (abs(t.ft0 - point.ft0) < abs(distance)) 
          distance = t.ft0 - point.ft0;
      }
      return distance;
    }

    Cluster() = default;
    Cluster(uint gps) : fGPSSecond(gps) { }
    Cluster(const Cluster& c) = default;

    template<class G>
    double
    GetTimeDifferenceInS(const G& g)
      const
    {
      return    g.fGPSSecond - fGPSSecond
             + (g.fMicroSecond - fMicroSecond)*1e-6;
    }

    double
    GetToTFraction()
      const
    {
      double tmp = 0;
      for (const auto& t : fT2s) {
        if (t.IsToT())
          ++tmp;
      }

      return tmp/fT2s.size();
    }

    template<class G>
    bool
    IsCompatible(const G& newGraph)
      const
    {
      const auto meanPosition = newGraph.GetAveragePosition();
      const double timeShift =   meanPosition.first*fu 
                               + meanPosition.second*fv;
      const double timeDifference = GetTimeDifferenceInS(newGraph)
                                    *kSecond*kMicroSecond;
      if (fabs(timeDifference - timeShift) 
            < kMaxTimeDifference)
        return true;

      //sanity check, avoid ambiguity due to lack of GPSs in T2Data
      if (fGPSSecond != newGraph.fGPSSecond)
        return false;

      for (const auto& t2 : fT2s) {
        if (newGraph.ContainsT2(t2))
          return true;
      }

      return false;
    }

    ClassDefNV(Cluster, 6);
  };
};
#endif