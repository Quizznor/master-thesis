#include <T2Dump/GraphSearch.h>

namespace t2 {

  void
  GraphSearch::AddT2(const T2Data& t2,
                     std::vector<Graph>& graphs,
                     const uint index)
  {
    fLastTriggerTime[t2.fId] = t2.fTime/kMicroSecond;
    fTimeOfLastTrigger = t2.fTime/kMicroSecond;

    GraphNode newNode(t2);
    newNode.fIndexInT2Array = index;
    newNode.fX = fStationInfos[t2.fId].fPosition.fX;
    newNode.fY = fStationInfos[t2.fId].fPosition.fY;

    auto itMatched = fLabeledNodes.end();

    for (auto it = fLabeledNodes.begin(); it != fLabeledNodes.end(); ++it) {
      if (it->back().fTime < newNode.fTime - kMaxTimeDifference/kMicroSecond) {
        if (it->size() < fMinimalNumberPerGraph &&
           !(it->size() == 2 && it->front().IsPlaneFrontDoublet(it->back()))) {  //noise case
          it = fLabeledNodes.erase(it);
          --it; //keep iterator in right place after deleting list entry
        } else {
          Graph g;
          g.fGPSSecond = fGPSSecond;
          const int beginningMus = it->front().fTime;
          const int endingMus = it->back().fTime;
          g.fMicroSecond = (beginningMus + endingMus)/2.;

          g.fNodesSignal.insert(g.fNodesSignal.end(),
                                        it->begin(), it->end());
          g.ComputeClosestCrowns();
          AddDeadNeighbours(g);
          g.CheckTimeAssignment();
          //decision if interesting or not may need triplets -> later
          graphs.push_back(g);

          it = fLabeledNodes.erase(it);
          --it; //see above
        }
      } else {
        //iterate backwards, as end are closer in time
        for (auto ritVector = it->rbegin(); ritVector != it->rend(); ++ritVector) {
          if (newNode.IsConnected(*ritVector, fOnlyCoherent, fMaxCrown, fMaxDeltaTLightning)) {
            if (itMatched == fLabeledNodes.end()) {
              it->push_back(newNode);
              itMatched = it;
            } else {  //join connected graphs
              itMatched->insert(itMatched->end(), it->begin(), it->end());
              std::sort(itMatched->begin(), itMatched->end());
              it = fLabeledNodes.erase(it);
              --it; //see above
            }
            break;
          }
        }
      }//still within 50 us
    }//iteration over known blocks

    if (itMatched == fLabeledNodes.end()) {
      std::vector<GraphNode> tmp;
      tmp.push_back(newNode);
      fLabeledNodes.push_front(tmp);
    }
  }


  void
  GraphSearch::AddDeadNeighbours(Graph& g)
    const
  {
    for (const auto& n : g.fNodesSignal) {
      for (const auto& id : fNeighbours[n.fId]) {
        if (IsDead(id)) {
          const auto& p = fStationInfos[id].fPosition;
          if (!g.ContainsId(id))
            g.fDeadNeighbours.emplace_back(n.fTime, id, 0, p.fX, p.fY);
        }
      }
    }
  }


  bool
  GraphSearch::IsDead(const ushort id)
    const
  {
    return IsDead(id, fTimeOfLastTrigger * kMicroSecond);
  }


  bool
  GraphSearch::IsDead(const ushort id, const int timeInMeter)
    const
  {
    const bool validStation = fStationInfos[id].IsValid(fGPSSecond);
    if (   timeInMeter / kMicroSecond - fLastTriggerTime[id] > kOneSecond/2
        && validStation) { //only a valid station is dead (e.g. moved aren't)
      return true;
    } else {
      return false;
    }
    return true;
  }


  void
  GraphSearch::ComputeNeighbours(const std::vector<char>& stationMask)
  {
    std::vector<ushort> tmp;
    for (uint i = 0; i < fStationInfos.size(); ++i)
      fNeighbours.push_back(tmp);

    for (uint id1 = 0; id1 < fStationInfos.size(); ++id1) {
      if (!stationMask[id1])
        continue;
      const auto n1 = GraphNode(id1,
                                fStationInfos[id1].fPosition.fX,
                                fStationInfos[id1].fPosition.fY);
      for (uint id2 = 0; id2 < fStationInfos.size(); ++id2) {
        if (id1 == id2 || !stationMask[id2])
          continue;
        const auto n2 = GraphNode(id2,
                                  fStationInfos[id2].fPosition.fX,
                                  fStationInfos[id2].fPosition.fY);
        if (n1.GetCrown(n2) == 1)
          fNeighbours[id1].push_back(id2);
      }
    }
  }


  void
  GraphSearch::SubstractOneSecond()
  {
    for (auto it = fLabeledNodes.begin(); it != fLabeledNodes.end(); ++it) {
      for (auto& n : *it) {
        n.fTime -= kOneSecond;
      }
    }

    for (auto& t : fLastTriggerTime)
      t -= kOneSecond;

    fTimeOfLastTrigger -= kOneSecond;
  }


  void
  GraphSearch::EndAnalysis(std::vector<Graph>& graphs)
  {
    for (auto it = fLabeledNodes.begin(); it != fLabeledNodes.end(); ++it) {
      if (it->size() < fMinimalNumberPerGraph &&
          !(it->size() == 2 && it->front().IsPlaneFrontDoublet(it->back())))
        continue;

      Graph g;
      g.fGPSSecond = fGPSSecond;
      const int beginningMus = it->front().fTime;
      const int endingMus = it->back().fTime;
      g.fMicroSecond = (beginningMus + endingMus)/2.;

      g.fNodesSignal.insert(g.fNodesSignal.end(),
                                    it->begin(), it->end());
      g.ComputeClosestCrowns();
      graphs.push_back(g);  //decision if interesting or not needs triplets -> later
    }
  }


  bool
  GraphSearch::IsInterestingGraph(const std::vector<GraphNode>& data)
    const
  {
    if (data.size() > fMinimalNumberPerGraph + 2)
      return true;

    int nSameId = 0;

    for (const auto& gn : data) {
      --nSameId;
      for (const auto& n : data) {
        if (n.fId == gn.fId)
          ++nSameId;
      }
    }
    if (nSameId > 0) {
      if (data.size() - nSameId > fMinimalNumberPerGraph)
        return true;
    }

    return false;
  }


  int
  GraphSearch::GetCurrentNumberOfNodes()
    const
  {
    int n = 0;
    for (auto it = fLabeledNodes.begin(); it != fLabeledNodes.end(); ++it) {
      n += it->size();
    }

    return n;
  }


  int
  GraphSearch::GetCurrentNumberOfSubGraphs()
    const
  {
    return fLabeledNodes.size();
  }


  std::vector<ushort>
  GraphSearch::GetNeighbours(const ushort id)
    const
  {
    if (id < fNeighbours.size()) {
      return fNeighbours[id];
    } else {
      return std::vector<ushort>();
    }
  }


  std::vector<ushort>
  GraphSearch::GetDeadNeighbours(const ushort id)
    const
  {
    std::vector<ushort> output;
    for (const auto& i : GetNeighbours(id)) {
      if (IsDead(i))
        output.push_back(i);
    }
    return output;
  }


  std::vector<ushort>
  GraphSearch::GetDeadNeighbours(const T2Data& t2)
    const
  {
    std::vector<ushort> output;
    for (const auto& i : GetNeighbours(t2.fId)) {
      if (IsDead(i, t2.fTime))
        output.push_back(i);
    }
    return output;
  }


  void
  GraphSearch::FillDeadStations(Cluster& c)
    const
  {
    for (const auto& t : c.fT2s) {
      const auto dn = GetDeadNeighbours(t);
      for (const auto& i : dn) {
        c.fDeadNeighbours.emplace_back(c.fMicroSecond, i, 0,
                                     fStationInfos[i].fPosition.fX,
                                     fStationInfos[i].fPosition.fY);
      }
    }
  }
};