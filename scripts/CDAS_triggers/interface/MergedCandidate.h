#ifndef _MergedCandidate_h_
#define _MergedCandidate_h_

#include <Rtypes.h>
#include <t2/T2Data.h>
#include <T2Dump/Triplet.h>
#include <vector>
#include <T2Dump/Utl.h>
#include <t2/StationInfo.h>
#include <T2Dump/GraphNode.h>
#include <interface/Graph.h>
#include <interface/t2Cluster.h>
#include <interface/Events.h>
#include <t3/T3Data.h>


enum class EventType {
  Undefined,
  SingleCluster,
  SingleGraph,
  MultiGraph,
  MultiCluster,
  Combined,
  T3,
  MultiT3 //2 or more separate formations each a T3, close in time
};

namespace t2 {

  //kind of a legacy name:
  // meant to merge Graph based analysis (compact signals)
  // with the Hough-transform like cluster search
  // into a single event-Candidate class
  // -> classification of this, (for background comparison)
  //  needs to be investigated still
  struct MergedCandidate {
    uint fGPSSecond = 0;
    int fMicroSecond = 0;

    std::vector<Cluster> fCluster;
    std::vector<Graph> fGraphs;

    // save triplets, that are formed by T2 of seperated
    // graphs. Potential improvement of Bg-rejection
    //  -> IsInteresting() [...] fGraphs.size() > 1 [...]
    std::vector<rTriplet> fMultiGraphTriplets;

    std::vector<GraphNode> fCommonT2s; //in >= 1 Graph && >= 1 Cluster
    std::vector<ClusterPoint> fCommonClusterPoints;

    // Noise refers to clusterpoints or T2s not assigned to signal parts
    // but close in time
    std::vector<ClusterPoint> fClusterNoise;
    std::vector<GraphNode> fNoiseT2s;   //can contain 'event' triggers
                                        // as there is no T1 information
                                        // -> 'lonely' stations

    EventType fType = EventType::Undefined;

    //for testing purposes, can be calculated from other data
    int fMinCrown = -1;
    float fToTFraction = -1;
    int fnT2s = 0;

    float fPlaneQuality = -1;

    int fNDoubleTrigger = -1;        //> init with -1 => indicate not processed parts
    double fPointSourceSpread = -1;  //> TODO
    int fNWideTriggers = -1;         //> TODO but init with -2. Set to -2 if before 27/10/16...

    MergedCandidate() = default;
    MergedCandidate(const Cluster& c);
    MergedCandidate(const Graph& g);
    MergedCandidate(const MergedCandidate& m) = default;

    //helper methods
    bool operator<(const MergedCandidate& m)
      const
    {
      return double(fGPSSecond) - double(m.fGPSSecond)
            + 1e-6*(double(fMicroSecond) - double(m.fMicroSecond)) < 0;
    }
    bool IsCompatibleToGraphs(const Graph&) const;
    bool IsCompatibleToGraphs(const Cluster&) const;
    bool IsCompatibleToCluster(const Graph&) const;
    bool IsCompatibleToCluster(const Cluster&) const;
    bool CheckTripletCompatibility(const rTriplet& t) const;
    bool IsSampledAnalysis() const;

    double GetEventTime() const;
    double GetEventTimeDifference(const MergedCandidate& m) const;
    int GetEventId() const; //NOTE: this is just a heuristic hack for now...
                            // essentially returns the microsecond, because by
                            // construction of the merging this should be unique
                            // within a single GPS second.

    void UpdateTimeEstimate();
    bool ContainsT2(const T2Data&) const;
    bool ContainsTriplet(const rTriplet&) const;

    void FindCommonT2s();
    void FindCommonClusterPoints();

    //methods for adding data
    bool AddCluster(const Cluster& c);
    bool AddGraph(const Graph& g);
    void AddNoise(const std::vector<T2Data>& t2s,
                  const std::vector<rTriplet>& recoTriplets);
    void AddMultiGraphTriplets(const std::vector<rTriplet>& recoTriplets);

    //`high' level methods
    bool IsInteresting();
    bool FileOutput() const;  //meant to decide whether an event-cand. is 'worthy' to be written to file
    std::string GenerateKey() const;
    std::vector<std::string> GenerateKeys() const;

    template<typename T>
    int
    MinimalCrownInCluster(const std::vector<t2::StationInfo<T>>& stationInfo)
      const
    {
      int minCrown = 100;

      for (const auto& c : fCluster) {
        for (uint i = 0; i < c.fT2s.size(); ++i) {
          for (uint j = i + 1; j < c.fT2s.size(); ++j) {
            const auto& pi = stationInfo[c.fT2s[i].fId].fPosition;
            const auto& pj = stationInfo[c.fT2s[j].fId].fPosition;

            const int tmp = crown(pi.fX, pj.fX, pi.fY, pj.fY);
            if (tmp < minCrown)
              minCrown = tmp;
          }
        }
      }

      return minCrown < 100 ? minCrown : -1;
    }
    /*template<typename T>
    void
    FillDevFromPlane(const std::vector<t2::StationInfo<T>>& stationInfo,
                     double tolerance = 3.)
    {
      for (auto& c : fCluster)
        c.FillDeviationsFromPlaneFront(stationInfo, tolerance);
    }*/

    //meant to distinguish different types of
    // events (standard, inclined, SD-rings, ...)
    // might include additional parameters e.g. number of T2s
    // -> only prototype up to now
    int GetEventClass() const;
    int AssignEventClass(bool ignoreT3 = false);
    bool ContainsT3() const;
    bool ContainsT3(const std::vector<StationInfo<long int>>&) const;
    bool ContainsCluster() const { return fCluster.size(); }

    uint GetNumberT2sGraphs() const;
    uint GetNumberT2sCluster() const;
    //uint GetNumberOfOutliers() const; //FillDevFromPlane has to be called beforehand !
    double GetNExpectedPointsInCluster() const; //combinatorics: Binomial (nT2 over 3)
    uint GetNPointsInCluster() const;
    uint GetMaxSizeOfGraph() const;
    uint GetNumberOfT2s() const;
    uint GetNumberTriplets() const;
    uint GetNumberOfDeadStations() const;
    int GetNWideTriggers() const;

    std::vector<T2Data> GetT2s(const bool addNoise = false) const;
    std::vector<T2Data> GetT2s(const int id, const bool addNoise = false,
                               const bool convertToMeter = true) const;
    std::vector<T2Data> GetCommonT2s() const;
    std::vector<T2Data> GetGraphT2s() const;    // exclusive w.r.t common t2s
    std::vector<T2Data> GetClusterT2s() const;  //  - '' -
    std::vector<T2Data> GetNoiseT2s() const;

    std::vector<GraphNode> GetGraphNodes() const;

    double GetToTFraction() const;
    double GetPhi() const;
    double GetTheta() const;
    double GetPlaneFrontQuality() const;

    int GetTimeLength() const;

    std::vector<ushort> GetIds() const;

    double CalculatedPointSourceVariance();
    int CountDoubleTriggers(const double timeWindow = 25/*us*/);

    void FillInformation(const std::vector<T2Data>& t2s,
                         const std::vector<rTriplet>& recoTriplets);

    //clustering correct, however information about T2s is missing!
    void RerunClustering(std::vector<Cluster>& output,
                         Cluster& noise,
                         double epsilon,
                         int mPts,
                         double truncationLimit = 0.) const;
    //returns bool(number of new clusters)
    bool RerunClustering(double epsilon,
                         int mPts,
                         double truncationLimit = 0.) const;

    ClassDefNV(MergedCandidate, 7);
  };


  inline
  std::ostream&
  operator<<(std::ostream& os, const MergedCandidate& m)
  {
    os << "Candidate[" << m.fGPSSecond
       << ", " << m.fMicroSecond
       << ", " << m.GetEventClass()
       << ", " << m.fGraphs.size()
       << ", " << m.fCluster.size();
    if (m.fType == EventType::T3) {
      for (const auto& n : m.fGraphs.front().fNodesSignal)
        os << ", " << n;
    }
    os << " nT2: " << m.GetNumberOfT2s()
       << "]";
    return os;
  }


  struct MCMatchingCandidate : public MergedCandidate {
    MCEvent fMCEvent;

    MCMatchingCandidate() = default;
    MCMatchingCandidate(const MergedCandidate& m)
      : MergedCandidate::MergedCandidate(m) {}

    ClassDefNV(MCMatchingCandidate, 1);
  };

};
#endif