#ifndef _T2candidates_h_
#define _T2candidates_h_

#include <Rtypes.h>
#include <t2/T2Data.h>
#include <T2Dump/Triplet.h>
#include <vector>
#include <algorithm>
#include <T2Dump/Utl.h>
#include <t2/StationInfo.h>
#include <T2Dump/GraphNode.h>
#include <interface/Events.h>
#include <interface/Graph.h>
#include <interface/t2Cluster.h>

typedef unsigned int uint;

namespace t2 {
  struct EventCandidate {
    uint fGPSSecond = 0;
    int fMicroSecond = 0;

    std::vector<Cluster> fCluster;
    Cluster fNoise;

    int fTotalTriplets = 0;

    void
    SetMicroSecond()
    {
      int n = 0;
      fMicroSecond = 0;
      fTotalTriplets = 0;
      for (const auto& cluster : fCluster) {
        fMicroSecond += cluster.fMicroSecond;
        fTotalTriplets += cluster.fData.size();
        ++n;
      }
      fMicroSecond /= n;
      fTotalTriplets += fNoise.fData.size();
    }

    ClassDefNV(EventCandidate, 2);
  };

  //adds triplets and `noise' T2s to a graph and implements method
  // to determine interesting events
  struct ConnectedGraphEventCandidate : public Graph { 
    std::vector<GraphNode> fNodesNoise;
    std::vector<rTriplet> fTriplets;

    //basic function to decide wether something is an
    // event candidate
    bool
    IsInteresting()
    {
      if (ContainsT3())
        return true;

      if ((fNodesSignal.size() == 5 && fMeanCrown < 2.)
          || fNodesSignal.size() > 5)
        return true;

      if (fNodesSignal.size() == 3 && fabs(fMeanCrown - 1) < .1) {
        if (!ContainsZeroCrown())
          return CheckForCorrespondingTriplet();
        else 
          return false;
      }

      if (fNodesSignal.size() == 4 && fMeanCrown > 0.9 && fMeanCrown < 1.4)
        return CheckForCorrespondingTriplet();

      return false;
    }

    //meant to check if in the 3 T2 case a plane front is reconstructable
    // additional exception case: aligned triggers are used, but not
    // reconstructed in the triplet analysis
    bool
    CheckForCorrespondingTriplet()
      const
    {
      for (const auto& t : fTriplets) {
        if (CheckTripletCompatibility(t))
          return true;
      }

      if (fNodesSignal.size() == 3) {
        const t2::Vector<int> delta1(fNodesSignal[0].fX - fNodesSignal[1].fX,
                                     fNodesSignal[0].fY - fNodesSignal[1].fY,
                                     0);
        const t2::Vector<int> delta2(fNodesSignal[0].fX - fNodesSignal[2].fX,
                                     fNodesSignal[0].fY - fNodesSignal[2].fY,
                                     0);
        if (abs(delta1*delta2) >= 0.99*delta1.XYMag()*delta2.XYMag()) 
          return CheckAlignedCompatibility();
      }

      return false;
    }

    ClassDefNV(ConnectedGraphEventCandidate, 3);
  };

  struct JoinedEventCandidate : public EventCandidate {
    MCEvent fMCEvent;

    JoinedEventCandidate() = default;
    JoinedEventCandidate(const EventCandidate& ec) : EventCandidate::EventCandidate(ec) {}

    ClassDefNV(JoinedEventCandidate, 1);
  };

  struct JoinedGraphCandidate : public ConnectedGraphEventCandidate {
    MCEvent fMCEvent;

    JoinedGraphCandidate() = default;
    JoinedGraphCandidate(const ConnectedGraphEventCandidate& ce) 
      : ConnectedGraphEventCandidate::ConnectedGraphEventCandidate(ce) { } 

    ClassDefNV(JoinedGraphCandidate, 1);
  };
};
#endif