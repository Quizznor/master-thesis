#ifndef _GraphSearch_
#define _GraphSearch_

#include <T2Dump/GraphNode.h>
#include <T2Dump/Units.h>

#include <t2/T2Data.h>
#include <t2/StationInfo.h>

#include <interface/T2Candidates.h>
#include <interface/t2Cluster.h>
#include <interface/Graph.h>

#include <io/RootOutFile.h>

#include <utility>
#include <algorithm>
#include <vector>
#include <list>
#include <string>


//a class that should implement the search for connected
// components in the graph (set of points) defined by the t2s
// idea is to stay iterative by simply adding the next
// (time ordered) t2

namespace t2 {


  class GraphSearch {
  private:
    std::vector<StationInfo<long>> fStationInfos;
    std::vector<std::vector<ushort>> fNeighbours;

    std::list<std::vector<GraphNode>> fLabeledNodes;  //list of compact formations of T2s
    int fLabel = 1;

    std::vector<int> fLastTriggerTime;
    int fTimeOfLastTrigger = 0;

    bool IsInterestingGraph(const std::vector<GraphNode>& data) const;
    void ComputeNeighbours(const std::vector<char>& stationMask); //neighbour == crown 1 station (i.e. problems with infill!)
    void AddDeadNeighbours(Graph& g) const;

  public:
    uint fMinimalNumberPerGraph = 3;
    uint fMaxCrown = 4;
    uint fGPSSecond = 0;
    bool fOnlyCoherent = false;   //exception for thunderstorm parts (50 us for nn)
    double fMaxDeltaTLightning = kMaxLightningSearchTimeDifference;
    std::vector<int> fAllowedGridTypes;

    void SubstractOneSecond();
    bool IsDead(const ushort id) const;
    bool IsDead(const ushort id, const int time) const;

    std::vector<ushort> GetNeighbours(const ushort id) const;
    std::vector<ushort> GetDeadNeighbours(const ushort id) const;
    std::vector<ushort> GetDeadNeighbours(const T2Data& t2) const;
    void FillDeadStations(Cluster& c) const;

    int GetCurrentNumberOfNodes() const;
    int GetCurrentNumberOfSubGraphs() const;

    void EndAnalysis(std::vector<Graph>& graphs); //write clusters still in memory

    template<typename T>
    void
    SetStationInfo(const std::vector<StationInfo<T>>& stationInfo,
                   const std::vector<char>& stationMask)
    {
      for (uint i = 0; i < stationInfo.size(); ++i)
        fStationInfos.push_back(stationInfo[i]);
      ComputeNeighbours(stationMask);
      fLastTriggerTime.resize(stationInfo.size(), 0);
    }

    void
    AddT2(const T2Data& t2,
          std::vector<Graph>& graphs,
          uint index); //for 'fast access'

    GraphSearch() : fLastTriggerTime(2000, 0) {}
    ~GraphSearch() = default;
    GraphSearch(const GraphSearch& b) = default;
  };

}

#endif