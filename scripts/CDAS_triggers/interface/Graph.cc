#include <interface/Graph.h>

ClassImp(t2::Graph)

namespace t2 {


  void
  Graph::CheckTimeAssignment()
  {
    if (0 <= fMicroSecond && fMicroSecond< 1e6)
      return;

    int dt = 0;
    if (fMicroSecond < 0) {
      --fGPSSecond;
      dt = 1e6;
    } else {
      ++fGPSSecond;
      dt = -1e6;
    }

    fMicroSecond += dt;

    for (auto& n : fNodesSignal)
      n.fTime += dt;
  }


  uint
  Graph::size()
    const
  {
    uint tmp = 0;
    for (const auto& n : fNodesSignal) {
      if (n)
        ++tmp;
    }
    return tmp;
  }


  bool
  Graph::ContainsId(const ushort id)
    const
  {
    for (const auto& n : fNodesSignal) {
      if (n.fId == id)
        return true;
    }

    for (const auto& n : fDeadNeighbours) {
      if (n.fId == id)
        return true;
    }
    return false;
  }


  bool
  Graph::ContainsT2(const T2Data& t2)
    const
  {
    return std::count(fNodesSignal.begin(), fNodesSignal.end(), t2);
  }


  bool
  Graph::ContainsNode(const GraphNode& n)
    const
  {
    return std::count(fNodesSignal.begin(), fNodesSignal.end(), n);
  }


  int
  Graph::GetNumberOfTripletCombinations()
    const
  {
    const uint s = size();
    return s*(s - 1)*(s - 2)/6;
  }


  //checks the graph by searching the nearest neighbour of
  // each node
  void
  Graph::ComputeClosestCrowns()
  {
    for (const auto& gn : fNodesSignal) {
      int crownMin = 100;
      if (!gn)
        continue;
      for (const auto& node2 : fNodesSignal) {
        if (node2 == gn || !node2)
          continue;
        const int crown = gn.GetCrown(node2);
        if (!crown) {
          crownMin = 0;
          break;
        } else if (crown < crownMin) {
          crownMin = crown;
        }
      }
      fMeanCrown += crownMin;
      fClosestCrown.push_back(crownMin);
    }
    fMeanCrown /= fClosestCrown.size();
  }


  //ignores crown = 0
  int
  Graph::GetMinimalCrown(bool useZeroCrowns)
    const
  {
    int minCrown = 100;
    for (const auto& i : fClosestCrown) {
      if (i < minCrown && (i || useZeroCrowns))
        minCrown = i;
    }

    return minCrown;
  }


  //checks if two triggers of the same station are used
  // (important for 3 node graphs)
  // note: in MC it can happen, that different ids are on the same position
  //       at the same time -> use position not id
  bool
  Graph::ContainsZeroCrown()
    const
  {
    for (const auto& crown : fClosestCrown) {
      if (!crown)
        return true;
    }

    return false;
  }


  //checks if some of the graph nodes make up the rTriplet t
  bool
  Graph::CheckTripletCompatibility(const rTriplet& t)
    const
  {
    if (abs(t.ft0 - fMicroSecond) > 500.)
      return false;

    short nTriggerInNode[3] = {0, 0, 0};

    for (const auto& node : fNodesSignal) {
      if (!node)
        continue;
      for (int i = 0; i < 3; ++i) {
        if (node == t.fTrigger[i])
          ++nTriggerInNode[i];
      }
      if (nTriggerInNode[0] && nTriggerInNode[1] && nTriggerInNode[2])
        return true;
    }
    return false;
  }


  //checks if the T2s of this candidate form a CDAS T3
  // ignore timing refers to skipping the dt < 5*crown + 3 condition
  //return value (work in progress): 0 - no T3; 1 - 3ToT ; 2 - 4T2
  int
  Graph::ContainsT3(bool print, bool ignoreTiming)
    const
  {
    //speed up
    if (fNodesSignal.size() < 3)
      return 0;
    if (fNodesSignal.size() < 4 &&
        GetToTFraction()*fNodesSignal.size() < 2.1)
      return 0;

    //construct `real' T3s: 3ToT-2C_1&3C_2 and 4T2-2C1&3C2&4C4 modes
    for (const auto& node1 : fNodesSignal) {
      if (!node1)
        continue;
      int nNeighboursToT[2] = {0, 0};         //count stations in C1, C2
      int nNeighboursAny[4] = {0, 0, 0, 0};   // ... C1, C2, C3, C4
      std::vector<uint> fUsedIds;

      for (const auto& node2 : fNodesSignal) {
        if (node1 == node2 || !node2)
          continue;

        const int crownCDAS = node1.GetCrown(node2);
        const int deltaT = (node1.fTime - node2.fTime) * kMicroSecond;

        if (crownCDAS > 4 || crownCDAS < 1)
          continue;

        //check compactness and timing conditions
        // time condition is (3 + 5*n) mu s, in CDAS XbAlgo.cc (<= 5*neighbour + dtime)
        // with dtime = 3
        //directly remove out of time candidates
        if (!ignoreTiming && abs(deltaT) > (5*crownCDAS + kJitter)*kMicroSecond)
          continue;

        if (std::find(fUsedIds.begin(), fUsedIds.end(), node2.fId) != fUsedIds.end())
          continue;

        if (node2.IsToT()) {
          for (int i = crownCDAS - 1; i < 2; ++i)
            ++nNeighboursToT[i];
        }

        for (int i = crownCDAS - 1; i < 4; ++i)
          ++nNeighboursAny[i];

        fUsedIds.push_back(node2.fId);
      }

      if (print) {
        std::cout << nNeighboursToT[0] << ' '
                  << nNeighboursToT[1] << " | ";
        for (int i = 0; i < 4; ++i)
          std::cout << nNeighboursAny[i] << ' ';
        std::cout << "(" << node1.IsToT() << ")" << std::endl;
      }
      //3ToT trigger (using, that the center is triggered by construction)
      if (node1.IsToT()
          && nNeighboursToT[0] >= 1
          && nNeighboursToT[1] >= 2) {
        return 2;
      }

      //4T2 mode 2C1&3C2&4C4
      if (nNeighboursAny[0] >= 1
          && nNeighboursAny[1] >= 2
          && nNeighboursAny[3] >= 3) {
        return 1;
      }
    }

    return 0;
  }


  //T3 like condidtions: condition[0]*C1 && condition[1]*C2 && ...
  // nCondition is the size of the condition array
  bool
  Graph::CheckCompactness(int* condition, int nCondidtion)
    const
  {
    for (const auto& node1 : fNodesSignal) {
      std::vector<int> nInCrown(nCondidtion, 0);

      if (!node1)
        continue;

      for (const auto& node2 : fNodesSignal) {
        if (node1 == node2 || !node2)
          continue;

        const int crownCDAS = node1.GetCrown(node2);
        const int deltaT = node1.fTime - node2.fTime;

        if (!crownCDAS) //same station
          continue;

        //check compactness and timing conditions
        // hexagons are approximated with circles -> overestimation of compactness
        // time condition is (3 + 5*n) mu s, in CDAS XbAlgo.cc (<= 5*neighbour + dtime)
        // with dtime = 3
        if (abs(deltaT) > kJitter + 5*crownCDAS) //directly remove out of time candidates
          continue;
        if (crownCDAS < nCondidtion + 1)
          for (int i = crownCDAS - 1; i < nCondidtion; ++i)
            ++nInCrown[i];
      }

      bool output = true;
      for (int i = 0; i < nCondidtion; ++i) {
        output = output && (condition[i] <= nInCrown[i]);
      }
      if (output)
        return true;
    }

    return false;
  }


  //T3 like condidtions: condition[0]*C1 && condition[1]*C2 && ...
  // nCondition is the size of the condition array
  bool
  Graph::CheckCompactness(const std::vector<int>& condition,
                          bool tot, bool print)
    const
  {
    if (print) {
      std::cout << "Testing condition:";
      for (const auto& a : condition)
        std::cout << " " << a;
      std::cout << " /condition";
    }

    for (const auto& node1 : fNodesSignal) {
      std::vector<int> nInCrown(condition.size(), 0);
      if ((tot && !node1.IsToT()) || !node1)
        continue;

      for (const auto& node2 : fNodesSignal) {
        if (node1 == node2 || !node2)
          continue;
        if (tot && !node2.IsToT())
          continue;

        const uint crownCDAS = node1.GetCrown(node2);
        const int deltaT = node1.fTime - node2.fTime;

        if (!crownCDAS) //same station
          continue;

        //check compactness and timing conditions
        // time condition is (3 + 5*n) mu s, in CDAS XbAlgo.cc (<= 5*neighbour + dtime)
        // with dtime = 3
        if (abs(deltaT) > kJitter + 5*crownCDAS) //directly remove out of time candidates
          continue;
        if (crownCDAS < condition.size() + 1)
          for (uint i = crownCDAS - 1; i < condition.size(); ++i)
            ++nInCrown[i];
      }

      if (print) {
        std::cout << " detected: ";
        for (const auto& a : nInCrown)
          std::cout << " " << a;
        std::cout << " /detected ";
      }

      bool output = true;
      for (uint i = 0; i < condition.size(); ++i) {
        output = output && (condition[i] <= nInCrown[i]);
      }
      if (output) {
        if (print)
          std::cout << " 1" << std::endl;
        return true;
      }
    }
    if (print)
      std::cout << " 0" << std::endl;
    return false;
  }


  //Get the index of the first (or last) station in
  // an aligned configuration, w.r.t. geometry not time
  // only for the case of 3 stations!
  //searches for the smallest x : y combination (x is priority)
  int
  Graph::FindGeomFirstOfAligned()
    const
  {
    if (size() != 3)
      return -1;  //should not be used in this case
    int min = 150000;
    int minY = 150000;
    int iMin = -1;

    for (int i = 0; i < 3; ++i) {
      if (abs(fNodesSignal[0].fX - fNodesSignal[1].fX) < 100) {
        if (minY > fNodesSignal[i].fY) {
          minY = fNodesSignal[i].fY;
          iMin = i;
        }
      } else {
        if (min > fNodesSignal[i].fX) {
          min = fNodesSignal[i].fX;
          iMin = i;
        }
      }
    }

    return iMin;
  }


  //check if an aligned 3 T2 configuration is
  // compatible with the arrival of front
  // (e.g. not `.' in time)
  bool
  Graph::CheckAlignedCompatibility()
    const
  {
    if (size() != 3)
      std::cerr << "warning: calling aligned-check not with 3 stations!" << std::endl;
    const int iFirstStation = FindGeomFirstOfAligned();

    if (iFirstStation < 0 || iFirstStation > 2)
      return false;

    double deltaT[2] = { 0, 0 };
    int dummy = 0;
    for (int i = 0; i < 3; ++i) {
      if (i == iFirstStation)
        continue;
      deltaT[dummy] = fNodesSignal[i].fTime - fNodesSignal[iFirstStation].fTime;
      const double dist = std::sqrt(pow(fNodesSignal[i].fX - fNodesSignal[iFirstStation].fX, 2)
                          + pow(fNodesSignal[i].fY - fNodesSignal[iFirstStation].fY, 2));
      deltaT[dummy++] /= dist;
    }

    if (  fabs(deltaT[0] - 2*deltaT[1]) < 0.01
       || fabs(deltaT[1] - 2*deltaT[0]) < 0.01)
      return true;

    return false;
  }


  bool
  Graph::IsPlaneFrontDoublet()
    const
  {
    if (size() == 2)
      return fNodesSignal.front().IsPlaneFrontDoublet(fNodesSignal.back());

    if (GetMinimalCrown() > 1)
      return false;

    for (const auto& n : fNodesSignal) {
      for (const auto& n2 : fNodesSignal) {
        if (n == n2)
          continue;
        const int crown = n.GetCrown(n2);
        if (crown != 1)
          continue;
        if (abs(n.fTime - n2.fTime) <= 5)
          return true;
      }
    }
    return false;
  }


  //checks whether all distances are compatible with speed of light
  bool
  Graph::IsWithinLightCones()
    const
  {
    for (const auto& n1 : fNodesSignal) {
      for (const auto& n2 : fNodesSignal) {
        if (&n1 == &n2)
          continue;
        const int crown = n1.GetCrown(n2);
        if (abs(n1.fTime - n2.fTime)*kMicroSecond >
            (5*crown + kJitter)*kMicroSecond)
          return false;
      }
    }
    return true;
  }


  //basic function to decide whether something is an
  // event candidate
  bool
  Graph::IsInteresting()
  {
    if (fUsedInMerging) //for non heuristic merging
      return false;

    if (ContainsT3()) {
      fIsT3 = true;
      return true;
    }

    if (fNodesSignal.size() > 5)
      return true;

    //`coherence'
    if (fAssociatedTriplets.size()) {
      return true;
    } else if (IsAlinged()) { //should this check timing?
      fIsAligned = true;
      return true;
    }

    fCompactness = GetConfiguration().first;
    if (fCompactness > 2300) //a.k.a 2C2 & 3C3
      return true;

    return IsPlaneFrontDoublet();
    //idea: use to identify rate of sub-threshold events
    // ('real' 2 station events)
  }


  //idea: all difference vectors are somewhat parallel
  //  => use scalar product with 0.95*abs(...) as condition
  bool
  Graph::IsAlinged()
    const
  {
    if (fNodesSignal.size() < 3)
      return false;

    std::vector<t2::Vector<int>> deltaXi;
    for (uint i = 0; i < fNodesSignal.size(); ++i) {
      for (uint j = i + 1; j < fNodesSignal.size(); ++j) {
        deltaXi.emplace_back(fNodesSignal[i].fX - fNodesSignal[j].fX,
                             fNodesSignal[i].fY - fNodesSignal[j].fY, 0);
      }
    }

    for (uint i = 0; i < deltaXi.size(); ++i) {
      for (uint j = i + 1; j < deltaXi.size(); ++j) {
        const double s = abs(deltaXi[i]*deltaXi[j]);
        const double s2 = 0.95*deltaXi[i].XYMag()*deltaXi[j].XYMag();
        if (s < s2)
          return false;
      }
    }
    return true;
  }


  //used to estimate potential relation between two graphs
  // or with a cluster
  std::pair<int, int>
  Graph::GetAveragePosition()
    const
  {
    utl::Accumulator::Mean x;
    utl::Accumulator::Mean y;

    for (const auto& n : fNodesSignal) {
      x(n.fX);
      y(n.fY);
    }

    return std::make_pair(x.GetMean(), y.GetMean());
  }


  double
  Graph::GetToTFraction()
    const
  {
    double tmp = 0;
    for (const auto& t : fNodesSignal) {
      if (t.IsToT())
        ++tmp;
    }

    return tmp/size();
  }


  std::pair<int, int>
  Graph::GetConfiguration(bool totOnly)  //a C1 & b C2 & c C3 & d C4;
    const             // more than C4 should not be relevant
  {
    int nToTMax = 0;
    int maxKey = 0;     // use abcd as key -> maximise to get the most compact form
    for (const auto& n1 : fNodesSignal) {
      int cN[4] = {1, 1, 1, 1};  // 1 as n1 is in C0
      int nToT = 0;
      if (n1.IsToT())
        ++nToT;

      if (totOnly && !n1.IsToT())
        continue;

      for (const auto& n2 : fNodesSignal) {
        if (&n1 == &n2)
          continue;
        if (totOnly && !n2.IsToT())
          continue;

        const int crown = n1.GetCrown(n2);
        if (!crown) // should 'usually' not happen
          continue;
        const int deltaT = n1.fTime - n2.fTime;
        if (abs(deltaT) > 3 + 5*crown) {
          continue;
        }

        if (crown < 5 && n2.IsToT())
          ++nToT;
        for (int i = crown - 1; i < 4; ++i)
          ++cN[i];
      }

      for (int i = 0; i < 4; ++i) {
        if (cN[i] > 10)
          std::cerr << "warning key ambigous!" << std::endl;
      }
      const int key = cN[0]*1000 + cN[1]*100 + cN[2]*10 + cN[3];
      if (key > maxKey) {
        maxKey = key;
        nToTMax = nToT;
      }
    }

    return std::make_pair(maxKey, nToTMax);
  }


  bool
  Graph::IsCompatible(const Graph& ng)
    const
  {
    const auto meanPositionsNew = ng.GetAveragePosition();
    const auto meanPosThis = GetAveragePosition();
    const double distance2 = utl::Sqr(double(meanPosThis.first
                                              - meanPositionsNew.first))
                             + utl::Sqr(double(meanPosThis.second
                                              - meanPositionsNew.second));
    const double absoluteTimeDiffInMeter =
                            GetAbsoluteTimeDifferenceInMeter(ng);
    if (distance2 + kToleranceDistance2
          > utl::Sqr(absoluteTimeDiffInMeter))
      return true;
    return false;
  }


  std::vector<T2Data>
  Graph::GetT2s()
    const
  {
    std::vector<T2Data> tmp;
    for (const auto& n : fNodesSignal) {
      if (n.fTriggers)
        tmp.emplace_back(n.fTime*kMicroSecond, n.fId, n.fTriggers);
    }

    return tmp;
  }


  uint
  Graph::GenerateKey()
    const
  {
    if (ContainsT3())
      return 0;

    return GetConfiguration().first;
  }


  //minimal crown n, from a triggered station,
  // that contains all other triggers of this pattern
  uint
  Graph::GetMinimalCrownOfPattern()
    const
  {
    utl::Accumulator::Min<uint> minMaximalCrown(100);
    for (const auto& n : fNodesSignal) {
      utl::Accumulator::Max<uint> maxCrown(0);
      for (const auto& n2 : fNodesSignal) {
        if (&n == &n2)
          continue;
        maxCrown(n.GetCrown(n2));
      }
      minMaximalCrown(maxCrown.GetMax());
    }
    return minMaximalCrown.GetMin();
  }


  // startnode: refers to the westernmost of the southernmost stations
  //  choice is arbitrary but unique
  uint
  Graph::GetIndexOfStartNode(const bool totOnly)
    const
  {
    std::pair<int, int> currentStart(60000, 60000); //should be well outside the array
    uint startIndex = fNodesSignal.size();

    for (uint i = 0; i < fNodesSignal.size(); ++i) {
      const auto& g = fNodesSignal[i];
      if (totOnly && !g.IsToT())
        continue;
      if (g.fY - currentStart.second < -500) {
        currentStart.first = g.fX;
        currentStart.second = g.fY;
        startIndex = i;
      } else if (abs(g.fY - currentStart.second) < 500) {
        if (g.fX + 1000 < currentStart.first) {
          currentStart.first = g.fX;
          currentStart.second = g.fY;
          startIndex = i;
        }
      }
    }
    return startIndex;
  }


  //tuple: decomposition (grid-vector a, grid-vector b), type: 0: th; 1- tot; -1-dead
  std::vector<std::tuple<int, int, int>>
  Graph::GetClassification(const bool totOnly, const bool includeNonWorking)
    const
  {
    std::vector<std::tuple<int, int, int>> output;
    const auto& startIndex = GetIndexOfStartNode(totOnly);
    if (startIndex >= fNodesSignal.size())
      return output;
    const auto& start = fNodesSignal[startIndex];
    int type = 0;
    if (start.IsToT())
      type = 1;
    else if (start.fTriggers)
      type = 0;
    else
      type = -1;
    output.push_back(std::make_tuple(0, 0, type));

    for (const auto& g : fNodesSignal) {
      if (&g == &start)
        continue;
      if (totOnly && !g.IsToT())
        continue;
      const t2::Vector<int> delta(g.fX - start.fX, g.fY - start.fY, 0);
      const auto& gridIndices = GetGridVectors(delta);
      int typeG = 0;
      if (g.IsToT())
        typeG = 1;
      else if (g.fTriggers)
        typeG = 0;
      else
        typeG = -1;
      output.push_back(std::make_tuple(gridIndices.first,
                                       gridIndices.second,
                                       typeG));
    }

    if (includeNonWorking) {
      for (const auto& g : fDeadNeighbours) {
        const t2::Vector<int> delta(g.fX - start.fX, g.fY - start.fY, 0);
        const auto& gridIndices = GetGridVectors(delta);
        output.push_back(std::make_tuple(gridIndices.first,
                                         gridIndices.second,
                                         -1));
      }
    }

    //sort (n, m) w.r.t. n^2 + m^2 (if equal with n)
    std::sort(output.begin(), output.end(),
       [](const std::tuple<int, int, int>& a, const std::tuple<int, int, int>& b)
          ->bool
        {
          const int sumA = std::pow(std::get<0>(a), 2) + std::pow(std::get<1>(a), 2);
          const int sumB = std::pow(std::get<0>(b), 2) + std::pow(std::get<1>(b), 2);

          if (sumA == sumB && std::abs(std::get<0>(a)) != std::abs(std::get<0>(b)))
            return std::abs(std::get<0>(a)) < std::abs(std::get<0>(b));
          else if (sumA == sumB)
            return std::abs(std::get<1>(a)) < std::abs(std::get<1>(b));
          return sumA < sumB;
        });

    return output;
  }


  std::string
  Graph::GetKey(const bool totOnly, const bool includeNonWorking)
    const
  {
    const auto& keys = GetClassification(totOnly);
    std::string outputValue = std::to_string(size());
    outputValue += "-" + std::to_string(int(IsWithinLightCones()));
    if (includeNonWorking) {
      const std::string containsDead = fDeadNeighbours.size() ? "1" : "0";
      outputValue += "-" + containsDead;
    }

    for (const auto& t : keys) {
      outputValue += "-(" + std::to_string(std::get<0>(t));
      outputValue += "," + std::to_string(std::get<1>(t)) + ","
                      + std::to_string(std::get<2>(t)) + ")";
    }
    return outputValue;
  }


  Graph
  Graph::ConstructRotatedGraph(int n60)
    const
  {
    Graph graph;

    if (n60 < 1)
      throw std::invalid_argument("no/negative rotations not implemented!");
    const t2::Vector<int> a(-750, 1299, 0);
    const t2::Vector<int> b(1500, 0, 0);

    const auto& classifier = GetClassification(false, true);

    for (const auto& t : classifier) {
      GraphNode tmp;
      tmp.fX = std::get<1>(t)*a.fX + (std::get<1>(t) - std::get<0>(t))*b.fX;
      tmp.fY = std::get<1>(t)*a.fY + (std::get<1>(t) - std::get<0>(t))*b.fY;
      const int triggerFlag = std::get<2>(t);
      switch (triggerFlag) {
        case (-1):
          tmp.fTriggers = 0;
          graph.fDeadNeighbours.push_back(tmp);
        break;
        case (0):
          tmp.fTriggers = 1;
          graph.fNodesSignal.push_back(tmp);
        break;
        case (1):
          tmp.fTriggers = 9;
          graph.fNodesSignal.push_back(tmp);
        break;
        default:
          std::cerr << "this should never happen (trigger type = " << triggerFlag << ")! "
                    << "-(" + std::to_string(std::get<0>(t))
                      + "," + std::to_string(std::get<1>(t)) + ","
                      + std::to_string(std::get<2>(t)) + ")"
                    << std::endl;
          throw std::runtime_error("abort ... fix the above!");

      }
    }

    if (n60 == 1)
      return graph;
    else
      return graph.ConstructRotatedGraph(n60 - 1);
  }


  Graph
  Graph::ConstructXMirroredGraph()
    const
  {
    Graph graph;
    const t2::Vector<int> a(-750, 1299, 0);
    const t2::Vector<int> b(1500, 0, 0);

    const auto& classifier = GetClassification(false, true);

    for (const auto& t : classifier) {
      GraphNode tmp;
      tmp.fX = std::get<1>(t)*a.fX + (std::get<1>(t) - std::get<0>(t))*b.fX;
      tmp.fY = std::get<1>(t)*a.fY + (std::get<1>(t) - std::get<0>(t))*b.fY;
      const int triggerFlag = std::get<2>(t);
      switch (triggerFlag) {
        case (-1):
          tmp.fTriggers = 0;
          graph.fDeadNeighbours.push_back(tmp);
        break;
        case (0):
          tmp.fTriggers = 1;
          graph.fNodesSignal.push_back(tmp);
        break;
        case (1):
          tmp.fTriggers = 9;
          graph.fNodesSignal.push_back(tmp);
        break;
        default:
          std::cerr << "this should never happen (trigger type = " << triggerFlag << ")! "
                    << "-(" + std::to_string(std::get<0>(t))
                      + "," + std::to_string(std::get<1>(t)) + ","
                      + std::to_string(std::get<2>(t)) + ")"
                    << std::endl;
          throw std::runtime_error("abort ... fix the above!");
      }
    }

    return graph;
  }


  std::vector<std::string>
  Graph::GetKeys(const bool includeNonWorking)
    const
  {
    std::vector<std::string> output;
    Graph mirrored = ConstructXMirroredGraph();

    output.push_back(GetKey(false, includeNonWorking));
    output.push_back(mirrored.GetKey(false, includeNonWorking));

    for (int i = 1; i < 6; ++i) {
      output.push_back(ConstructRotatedGraph(i).GetKey(false, includeNonWorking));
      output.push_back(mirrored.ConstructRotatedGraph(i).
                        GetKey(false, includeNonWorking));
    }
    return output;
  }


  //returns the alphabetically minimal key of
  // this graph making it a unique key for each pattern
  std::string
  Graph::GetMinKey(const bool includeNonWorking)
    const
  {
    utl::Accumulator::Min<std::string> minKey(GetKey(false, includeNonWorking));
    Graph mirrored = ConstructXMirroredGraph();

    minKey(mirrored.GetKey(false, includeNonWorking));

    for (int i = 1; i < 6; ++i) {
      minKey(ConstructRotatedGraph(i).GetKey(false, includeNonWorking));
      minKey(mirrored.ConstructRotatedGraph(i).GetKey(false, includeNonWorking));
    }
    return minKey.GetMin();
  }


  std::vector<std::vector<std::tuple<int, int, int>>>
  Graph::GetClassifications()
    const
  {
    std::vector<std::vector<std::tuple<int, int, int>>> output;
    Graph mirrored = ConstructXMirroredGraph();
    output.push_back(GetClassification());
    output.push_back(mirrored.GetClassification());

    for (int i = 1; i < 6; ++i) {
      output.push_back(ConstructRotatedGraph(i).GetClassification());
      output.push_back(mirrored.ConstructRotatedGraph(i).GetClassification());
    }
    return output;
  }

}