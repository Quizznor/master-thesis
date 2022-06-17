#ifndef _t2Graph_h_
#define _t2Graph_h_

#include <Rtypes.h>
#include <t2/T2Data.h>
#include <T2Dump/GraphNode.h>
#include <vector>
#include <algorithm>
#include <utility>
#include <T2Dump/Utl.h>
#include <T2Dump/Triplet.h>
#include <utl/Accumulator.h>
#include <string>
#include <exception>
#include <tuple>

enum class T3Type {
  eNone = 0,
  e4T2 = 1,
  e3ToT = 2
};


namespace t2 {


  //a first guess for a class to bundle the information
  // present in a connected graph in t2s.
  struct Graph {
    uint fGPSSecond = 0;
    int fMicroSecond = 0;

    float fMeanCrown = 0;
    bool fIsT3 = false;

    bool fIsAligned = false;     //!
    int fCompactness = -1;       //!
    bool fUsedInMerging = false; //!  needed as merging is not strictly time ordered

    std::vector<short> fClosestCrown;
    std::vector<GraphNode> fNodesSignal;
    std::vector<GraphNode> fDeadNeighbours;

    std::vector<rTriplet> fAssociatedTriplets;
    //triplets made of graph nodes

    template<class C>
    bool
    operator<(const C& b)
      const
    {
      const double deltaT = (fGPSSecond - b.fGPSSecond) + (fMicroSecond - b.fMicroSecond)*1e-6;
      return deltaT < 0;
    }

    template<class G>
    double
    GetAbsoluteTimeDifferenceInMeter(const G& g)
      const
    {
      const double deltaT = (g.fGPSSecond - fGPSSecond)
                            + (g.fMicroSecond - fMicroSecond)*1e-6;
      return fabs(deltaT)*kOneSecond*kMicroSecond;
    }

    template<class G>
    double
    GetTimeDifferenceInS(const G& g)
      const
    {
      return  g.fGPSSecond - fGPSSecond
           + (g.fMicroSecond - fMicroSecond)*1e-6;
    }

    // avoid negative values of us in EventTime + Nodes
    void CheckTimeAssignment();

    uint size() const;
    double GetToTFraction() const;

    bool ContainsId(const ushort id) const;
    bool ContainsT2(const T2Data& t2) const;
    bool ContainsNode(const GraphNode& n) const;

    int GetNumberOfTripletCombinations() const;

    bool IsCompatible(const Graph& ng) const;

    //checks the graph by searching the nearest neighbour of each node
    void ComputeClosestCrowns();

    //ignores crown = 0
    int GetMinimalCrown(bool useZeroCrowns = false) const;

    //checks if two triggers of the same station are used
    // (important for 3 node graphs)
    // note: in MC it can happen, that different ids are on the same position
    //       at the same time -> use position not id
    bool ContainsZeroCrown() const;

    //checks if some of the graph nodes make up the rTriplet t
    bool CheckTripletCompatibility(const rTriplet& t) const;

    //checks if the T2s of this candidate form a CDAS T3
    // ignore timing refers to skipping the dt < 5*crown + 3 condition
    //return value (work in progress): 0 - no T3; 1 - 3ToT ; 2 - 4T2
    int ContainsT3(bool print = false, bool ignoreTiming = false) const;

    //T3 like condidtions: condition[0]*C1 && condition[1]*C2 && ...
    // nCondition is the size of the condition array
    bool CheckCompactness(int* condition, int nCondidtion) const;

    //T3 like condidtions: condition[0]*C1 && condition[1]*C2 && ...
    // nCondition is the size of the condition array
    bool CheckCompactness(const std::vector<int>& condition,
                          bool tot = false,
                          bool print = false) const;

    //Get the index of the first (or last) station in
    // an aligned configuration, w.r.t. geometry not time
    // only for the case of 3 stations!
    //searches for the smallest x : y combination (x is priority)
    int FindGeomFirstOfAligned() const;

    //check if an aligned 3 T2 configuration is
    // compatible with the arrival of front
    // (e.g. not `.' in time)
    bool CheckAlignedCompatibility() const;
    bool IsPlaneFrontDoublet() const;

    //checks whether all distances are compatible with speed of light
    bool IsWithinLightCones() const;

    //basic function to decide whether something is an
    // event candidate
    bool IsInteresting();

    //idea: all difference vectors are somewhat parallel
    //  => use scalar product with 0.95*abs(...) as condition
    bool IsAlinged() const;

    //used to estimate potential relation between two graphs
    // or with a cluster
    std::pair<int, int> GetAveragePosition() const;

    //a C1 & b C2 & c C3 & d C4;
    // more than C4 should not be relevant
    // returns Key, nToTMax
    std::pair<int, int> GetConfiguration(bool totOnly = false) const;

    std::vector<T2Data> GetT2s() const;
    uint GenerateKey() const;

    //minimal crown n, from a triggered station,
    // that contains all other triggers of this pattern
    uint GetMinimalCrownOfPattern() const;

    //startnode: refers to the westernmost  of the southernmost stations
    // choice is arbitrary but unique
    uint GetIndexOfStartNode(const bool totOnly = false) const;

    //tuple: decomposition (grid-vector a, grid-vector b), type: 0: th; 1- tot; -1-dead
    std::vector<std::tuple<int, int, int>>
    GetClassification(const bool totOnly = false,
                      const bool includeNonWorking = false) const;

    std::string
    GetKey(const bool totOnly = false, const bool includeNonWorking = false) const;

    Graph ConstructRotatedGraph(int n60 = 1) const;
    Graph ConstructXMirroredGraph() const;

    std::vector<std::string>
    GetKeys(const bool includeNonWorking = false) const;

    //returns the alphabetically minimal key of
    // this graph making it a unique key for each pattern
    std::string
    GetMinKey(const bool includeNonWorking = false) const;

    std::vector<std::vector<std::tuple<int, int, int>>>
    GetClassifications() const;

    ClassDefNV(Graph, 1);
  };


  inline
  std::ostream&
  operator<<(std::ostream& os, const Graph& g)
  {
    os << "(" << g.fGPSSecond << ", " << g.fMicroSecond << ": ";
    for (const auto& n : g.fNodesSignal)
      os << n << " | ";
    return os;
  }
};

#endif
