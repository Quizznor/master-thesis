#include <interface/MergedCandidate.h>
#include <utl/Accumulator.h>
#include <utl/Math.h>
#include <TMath.h>
#include <algorithm>
#include <exception>
#include <T2Dump/Utl.h>

ClassImp(t2::MergedCandidate)

namespace t2 {

  MergedCandidate::MergedCandidate(const Cluster& c) : fGPSSecond(0)
  {
    if (!AddCluster(c))
      throw std::logic_error("this should never happen");
  }


  MergedCandidate::MergedCandidate(const Graph& g) : fGPSSecond(0)
  {
    if (!AddGraph(g))
      throw std::logic_error("this should never happen");
  }


  //helper methods
  bool
  MergedCandidate::IsCompatibleToGraphs(const Graph& newGraph)
    const
  {
    for (const auto& g : fGraphs) {
      if (g.IsCompatible(newGraph))
        return true;
    }

    return false;
  }


  bool
  MergedCandidate::IsCompatibleToGraphs(const Cluster& newCluster)
    const
  {
    for (const auto& g : fGraphs) {
      if (newCluster.IsCompatible(g))
        return true;
    }

    return false;
  }


  bool
  MergedCandidate::IsCompatibleToCluster(const Graph& newGraph)
    const
  {
    for (const auto& c : fCluster) {
      if (c.IsCompatible(newGraph))
        return true;
    }

    return false;
  }


  bool
  MergedCandidate::IsCompatibleToCluster(const Cluster& newCluster)
    const
  {
    for (const auto& c : fCluster) {
      //kSecond = 1e6... for now (r697) correct, but unclear (!)
      if (fabs(c.GetTimeDifferenceInS(newCluster)*kSecond*kMicroSecond)
            < kMaxTimeDifference)
        return true;
    }

    return false;
  }


  bool
  MergedCandidate::CheckTripletCompatibility(const rTriplet& t)
    const
  {
    short nTriggerInNode[3] = {0, 0, 0};

    for (const auto& g : fGraphs) {
      for (const auto& node : g.fNodesSignal) {
        for (int i = 0; i < 3; ++i) {
          if (node == t.fTrigger[i])
            ++nTriggerInNode[i];
        }
        if (nTriggerInNode[0] && nTriggerInNode[1] && nTriggerInNode[2])
          return true;
      }
    }
    return false;
  }


  void
  MergedCandidate::UpdateTimeEstimate()
  {
    utl::Accumulator::Mean mMicroSecond;
    for (const auto& c : fCluster)
      mMicroSecond(c.fMicroSecond);
    for (const auto& g : fGraphs)
      mMicroSecond(g.fMicroSecond);

    fMicroSecond = mMicroSecond.GetMean();

    if (fMicroSecond > 1e6) {
      fMicroSecond -= 1e6;
      ++fGPSSecond;
    } else if (fMicroSecond < 0) {
      fMicroSecond += 1e6;
      --fGPSSecond;
    }
  }


  double
  MergedCandidate::GetEventTime()
    const
  {
    return fGPSSecond + 1e-6*fMicroSecond;
  }


  double
  MergedCandidate::GetEventTimeDifference(const MergedCandidate& m)
    const
  {
    return int(fGPSSecond) - int(m.fGPSSecond) + 1e-6*(fMicroSecond - m.fMicroSecond);
  }


  int
  MergedCandidate::GetEventId()
    const
  {
    return fMicroSecond;
  }


  bool
  MergedCandidate::ContainsT2(const T2Data& t2)
    const
  {
    if (std::find(fCommonT2s.begin(), fCommonT2s.end(), t2) != fCommonT2s.end())
      return true;

    for (const auto& c : fCluster) {
      if (std::find(c.fT2s.begin(), c.fT2s.end(), t2) != c.fT2s.end())
        return true;
    }

    for (const auto& g : fGraphs) {
      if (std::find(g.fNodesSignal.begin(), g.fNodesSignal.end(), t2)
          != g.fNodesSignal.end())
        return true;
    }

    return false;
  }


  bool
  MergedCandidate::ContainsTriplet(const rTriplet& t)
    const
  {
    for (const auto& c : fCluster) {
      if (std::count(c.fData.begin(), c.fData.end(), t))
        return true;
    }
    for (const auto& g : fGraphs) {
      if (std::count(g.fAssociatedTriplets.begin(),
                     g.fAssociatedTriplets.end(), t))
        return true;
    }
    for (const auto& triplet : fMultiGraphTriplets) {
      if (t == triplet)
        return true;
    }
    return false;
  }


  void
  MergedCandidate::FindCommonT2s()
  {
    for (const auto& g : fGraphs) {
      for (const auto& n : g.fNodesSignal) {
        bool foundMatch = false;
        for (const auto& c : fCluster) {
          for (const auto& t2 : c.fT2s) {
            if (n == t2) {
              if (!std::count(fCommonT2s.begin(), fCommonT2s.end(), n)) {
                fCommonT2s.push_back(n);
                foundMatch = true;
                break;
              }
            }
          }
          if (foundMatch)
            break;
        }
      }
    }
  }


  void
  MergedCandidate::FindCommonClusterPoints()
  {
    for (const auto& g : fGraphs) {
      for (const auto& t : g.fAssociatedTriplets) {
        bool foundMatch = false;
        for (const auto& c : fCluster) {
          for (const auto& t2 : c.fData) {
            if (t2 == t) {
              if (!std::count(fCommonClusterPoints.begin(),
                              fCommonClusterPoints.end(), t)) {
                fCommonClusterPoints.push_back(t);
                foundMatch = true;
                break;
              }
            }
          }
          if (foundMatch)
            break;
        }
      }
    }
  }


  //methods for adding data
  bool
  MergedCandidate::AddCluster(const Cluster& c)
  {
    if (!c.fGPSSecond)
      std::cerr << "this should never happen (c)" << std::endl;
    if (!fGPSSecond) {
      fGPSSecond = c.fGPSSecond;
      fMicroSecond = c.fMicroSecond;

      fCluster.push_back(c);
      UpdateTimeEstimate();
      return true;
    } else {
      if (!(IsCompatibleToCluster(c) ||
            IsCompatibleToGraphs(c)))
        return false;

      fCluster.push_back(c);
      UpdateTimeEstimate();

      return true;
    }
  }


  bool
  MergedCandidate::AddGraph(const Graph& g)
  {
    if (!g.fGPSSecond)
      std::cerr << "this should never happen (g)" << std::endl;

    if (!fGPSSecond) {
      fGPSSecond = g.fGPSSecond;
      fMicroSecond = g.fMicroSecond;

      fGraphs.push_back(g);
      UpdateTimeEstimate();
      return true;
    } else {
      if (!( IsCompatibleToCluster(g) ||
             IsCompatibleToGraphs(g) ))
        return false;

      fGraphs.push_back(g);
      UpdateTimeEstimate();

      return true;
    }
  }


  void
  MergedCandidate::AddNoise(const std::vector<T2Data>& t2s,
                            const std::vector<rTriplet>& recoTriplets)
  {
    const int timeTolerance = 100 * kMicroSecond;
    const int timeToleranceLow = 500 * kMicroSecond;
    //collecting t2s
    int microSecondOfFirstT2 = kOneSecond*kMicroSecond;
    int microSecondOfLastT2 = 0;

    for (const auto& g : fGraphs) {
      for (const auto& n : g.fNodesSignal) {
        if (n.fTime*kMicroSecond > microSecondOfLastT2)
          microSecondOfLastT2 = n.fTime*kMicroSecond;
        if (n.fTime*kMicroSecond < microSecondOfFirstT2)
          microSecondOfFirstT2 = n.fTime*kMicroSecond;
      }
    }

    for (const auto& c : fCluster) {
      for (const auto& t2 : c.fT2s) {
        if (t2.fTime > microSecondOfLastT2)
          microSecondOfLastT2 = t2.fTime;
        if (t2.fTime < microSecondOfFirstT2)
          microSecondOfFirstT2 = t2.fTime;
      }
    }

    const auto itEnd = t2s.end();
    const auto idVec = GetIds();

    for (auto it = t2s.begin(); it != itEnd; ++it) {
      // for filtering previous triggers of stations
      //  check of dead time
      if (it->fTime < microSecondOfFirstT2 - 1000 * kMicroSecond)
        continue;
      if (std::find(idVec.begin(), idVec.end(), it->fId) != idVec.end()) {
        if (!ContainsT2(*it))
          fNoiseT2s.push_back(*it);
        continue;
      }

      if (it->fTime < microSecondOfFirstT2 - timeToleranceLow)
        continue;
      if (it->fTime > microSecondOfLastT2 + timeTolerance)
        break;

      if (!ContainsT2(*it))
        fNoiseT2s.push_back(*it);
    }

    //collecting triplets
    const int t0Tolerance = 50.*kMicroSecond;  //us, but ft0 is also in us -> still to update
    int tZeroOfFirstSignalTriplet = kOneSecond*kMicroSecond;
    int tZeroOfLastSignalTriplet = -1000*kMicroSecond;

    for (const auto& g : fGraphs) {
      for (const auto& t : g.fAssociatedTriplets) {
        if (t.ft0*kMicroSecond > tZeroOfLastSignalTriplet)
          tZeroOfLastSignalTriplet = t.ft0*kMicroSecond;
        if (t.ft0*kMicroSecond < tZeroOfFirstSignalTriplet)
          tZeroOfFirstSignalTriplet = t.ft0*kMicroSecond;
      }
    }

    for (const auto& c : fCluster) {
      for (const auto& t : c.fData) {
        if (t.ft0*kMicroSecond > tZeroOfLastSignalTriplet)
          tZeroOfLastSignalTriplet = t.ft0*kMicroSecond;
        if (t.ft0*kMicroSecond < tZeroOfFirstSignalTriplet)
          tZeroOfFirstSignalTriplet = t.ft0*kMicroSecond;
      }
    }

    //if no triplet is associated, use absolute time to collect noise
    if (tZeroOfLastSignalTriplet == -1000*kMicroSecond) {
      tZeroOfFirstSignalTriplet = microSecondOfFirstT2;
      tZeroOfLastSignalTriplet = microSecondOfLastT2;
    }

    for (auto itTriplet = recoTriplets.begin();
              itTriplet != recoTriplets.end();
              ++itTriplet) {
      if (itTriplet->ft0*kMicroSecond
           < tZeroOfFirstSignalTriplet - t0Tolerance)
        continue;
      if (itTriplet->ft0*kMicroSecond
           > tZeroOfLastSignalTriplet + t0Tolerance)
        break;

      if (!ContainsTriplet(*itTriplet))
        fClusterNoise.emplace_back(*itTriplet);
    }
  }


  int
  MergedCandidate::GetTimeLength()
    const
  {
    int microSecondOfFirstT2 = kOneSecond*kMicroSecond;
    int microSecondOfLastT2 = 0;

    for (const auto& g : fGraphs) {
      for (const auto& n : g.fNodesSignal) {
        if (n.fTime*kMicroSecond > microSecondOfLastT2)
          microSecondOfLastT2 = n.fTime*kMicroSecond;
        if (n.fTime*kMicroSecond < microSecondOfFirstT2)
          microSecondOfFirstT2 = n.fTime*kMicroSecond;
      }
    }

    for (const auto& c : fCluster) {
      for (const auto& t2 : c.fT2s) {
        if (t2.fTime > microSecondOfLastT2)
          microSecondOfLastT2 = t2.fTime;
        if (t2.fTime < microSecondOfFirstT2)
          microSecondOfFirstT2 = t2.fTime;
      }
    }

    return (microSecondOfLastT2 - microSecondOfFirstT2)/kMicroSecond;
  }


  //should be used after the graphs have their associated triplets
  void
  MergedCandidate::AddMultiGraphTriplets(const std::vector<rTriplet>& recoTriplets)
  {
    //use the time ordering to jump to a good starting point wo scanning
    // should on avg. be a lot faster than linear search
    const int kTolerance = 300.;

    if (!recoTriplets.size())
      return;

    if (fGraphs.size() < 2)
      return;

    int startIndex = fMicroSecond/1e6*int(recoTriplets.size()); //int as counted backwards!
    if (startIndex > int(recoTriplets.size()))
      startIndex = recoTriplets.size() - 1;
    else if (startIndex < 0)
      startIndex = 0;

    for ( ; startIndex >= 0; --startIndex) {
      const auto& t = recoTriplets[startIndex];
      if (t.ft0 < fMicroSecond - kTolerance)
        break;
    }
    ++startIndex; //first valid triplet

    for (uint i = startIndex; i < recoTriplets.size(); ++i) {
      const auto& t = recoTriplets[i];
      if (t.ft0 > fMicroSecond + kTolerance)
        break;
      if (CheckTripletCompatibility(t)) {
        if (!ContainsTriplet(t))
          fMultiGraphTriplets.push_back(t);
      }
    }
  }


  bool
  MergedCandidate::ContainsT3()
    const
  {
    for (const auto& g : fGraphs) {
      if (g.ContainsT3())
        return true;
    }
    return false;
  }


  bool
  MergedCandidate::ContainsT3(const std::vector<StationInfo<long int>>& stationInfos)
    const
  {
    for (const auto& g : fGraphs) {
      if (g.ContainsT3())
        return true;
    }
    for (const auto& c : fCluster) {
      if (c.ContainsT3(stationInfos)) {
        std::cerr << "This should never happen, if merging works!"
                  << std::endl;
        return true;
      }
    }
    return false;
  }


  bool
  MergedCandidate::IsSampledAnalysis()
    const
  {
    for (const auto& c : fCluster) {
      if (c.fT2s.size() && c.fData.empty())
        return true;
    }
    return false;
  }


  //by construction different graphs have to be disjunct
  uint
  MergedCandidate::GetNumberT2sGraphs()
    const
  {
    int n = 0;
    for (const auto& g : fGraphs) {
      n += g.fNodesSignal.size();
    }

    return n;
  }


  //in contrast to graphs,
  // the same T2 can appear in multiple clusters
  // -> avoid double counting with checks (slower)
  uint
  MergedCandidate::GetNumberT2sCluster()
    const
  {
    std::vector<T2Data> tmp;

    for (const auto& c : fCluster) {
      for (uint i = 0; i < c.fT2s.size(); ++i) {
        const auto& t = c.fT2s[i];
        if (fCluster.size() == 1 || //avoid check if only one cluster
            std::find(tmp.begin(), tmp.end(), t) == tmp.end())
          tmp.push_back(t);
      }
    }

    return tmp.size();
  }


  int
  MergedCandidate::GetNWideTriggers()
    const
  {
    if (fGPSSecond < t2::kTimeTriggersUpdate)
      return -2;

    const auto t2s = GetT2s();
    uint nWide = 0;
    for (const auto& t2 : t2s) {
      if (t2.IsWide())
        ++nWide;
    }
    return nWide;
  }


  /*
  Concept: check how coherent the signal is
    by assuming a point source at the first station and calculating
    the variance around the expected times for the other stations.

    -if the station is off in time, doesn't matter because of variance
    -if station is not in the 'centre', the whole argument of this
    construction breaks down! -> not checked yet as this is
    prototyping!
  */
  double
  MergedCandidate::CalculatedPointSourceVariance()
  {
    const auto nodeVec = GetGraphNodes();  // this has already x,y information
                                           // and it anyway needs distinction only for
                                           // compact cases -> if its in a candidate
                                           // and not a graph, it's a different topology
    utl::Accumulator::Var var;
    const auto& firstTrigger = nodeVec.front();

    for (const auto& n : nodeVec) {
      if (n == firstTrigger)
        continue;
      const double r = std::sqrt(std::pow(n.fX - firstTrigger.fX, 2)
                                 + std::pow(n.fY - firstTrigger.fY, 2));
      const double tExpected = firstTrigger.fTime + r / t2::kMicroSecond;
      var(n.fTime - tExpected);
    }

    fPointSourceSpread = var.GetCount() > 1 ? var.GetVar() : 0;
    return fPointSourceSpread;
  }


  /*
  idea: In SD-'rings' there are long-signal stations.
    Thus having multiple T2s (2 mostly) directly after
    one-another is probably a good sign of SD-ring like
    signals.
    Here we take the candidate and count how many stations
    have a second T2 within the specified time window.
    The dead-time due to the trace length is 19.2 mus.
    Therefore we take 20 us + margin as default value.

    The default value of 25 is not optimised! it is chosen by feeling
    of the author on initial implementation!
  */
  int
  MergedCandidate::CountDoubleTriggers(const double timeWindow)
  {
    const auto& t2Vec = GetT2s();
    std::map<int, std::vector<t2::T2Data>> t2map;

    for (const auto& t2 : t2Vec) {
      auto it = t2map.find(t2.fId);
      if (it == t2map.end()) {
        std::vector<t2::T2Data> tmp;
        tmp.push_back(t2);
        t2map.insert(std::make_pair(t2.fId, tmp));
      } else {
        it->second.push_back(t2);
      }
    }
    int nDouble = 0;
    for (auto& key : t2map) {
      auto t2vec = key.second;
      if (t2vec.size() < 2)
        continue;

      std::sort(t2vec.begin(), t2vec.end());

      for (int i = 1, n = t2vec.size(); i < n; ++i) {
        const double dt = (t2vec[i].fTime - t2vec[i - 1].fTime) / t2::kMicroSecond;
        if (dt < timeWindow)
          ++nDouble;
      }
    }

    fNDoubleTrigger = nDouble;
    return fNDoubleTrigger;
  }

  /*uint
  MergedCandidate::GetNumberOfOutliers()
    const
  {
    uint n = 0;
    for (const auto& c : fCluster) {
      if (c.fsigma2T < 0) {
        std::cerr << "warning: Deviations From Plane not set!" << std::endl;
        continue;
      }
      n += c.fnOutlier;
    }

    return n;
  }*/


  double
  MergedCandidate::GetToTFraction()
    const
  {
    if (fCommonT2s.size()) {
      double tmp = 0;
      for (const auto& t : fCommonT2s) {
        if (t.IsToT())
          ++tmp;
      }
      return tmp/fCommonT2s.size();
    } else {
      utl::Accumulator::Mean tmp;
      for (const auto& g : fGraphs) {
        tmp(g.GetToTFraction());
      }

      for (const auto& c : fCluster) {
        tmp (c.GetToTFraction());
      }

      return tmp.GetMean();
    }
  }


  //meant to collect the 'signal' ids
  // in t3 candidates for the comparison
  // with the CDAS T3s. Loops only over
  // graph parts, as T3s cannot be generated
  // without a graph that contains this T3
  std::vector<ushort>
  MergedCandidate::GetIds()
    const
  {
    std::vector<ushort> tmp;
    for (const auto& g : fGraphs) {
      for (const auto& t2 : g.fNodesSignal) {
        if (std::find(tmp.begin(), tmp.end(), t2.fId) == tmp.end())
          tmp.push_back(t2.fId);
      }
    }

    return tmp;
  }


  //avoids double counting, as no commonClusterPoitns are used
  uint
  MergedCandidate::GetNumberTriplets()
    const
  {
    uint tmp = 0;
    for (const auto& c : fCluster) {
      tmp += c.fData.size();
    }

    for (const auto& g : fGraphs) {
      tmp += g.fAssociatedTriplets.size();
    }

    tmp += fMultiGraphTriplets.size();
    return tmp;
  }


  void
  MergedCandidate::RerunClustering(
      std::vector<Cluster>& output,
      Cluster& noise,
      double epsilon,
      int mPts,
      double truncationLimit)
    const
  {
    std::vector<rTriplet> tmp;

    for (const auto& c : fCluster) {
      for (const auto& p : c.fData)
        tmp.emplace_back(p);
    }

    for (const auto& g : fGraphs) {
      for (const auto& t : g.fAssociatedTriplets)
        tmp.emplace_back(t);
    }

    for (const auto& t : fMultiGraphTriplets)
      tmp.emplace_back(t);

    for (const auto& p : fClusterNoise)
      tmp.emplace_back(p);

    std::sort(tmp.begin(), tmp.end());
    output.clear();
    DBScan(tmp, epsilon, mPts, output, noise, truncationLimit);
  }


  bool
  MergedCandidate::RerunClustering(double epsilon,
                                   int mPts,
                                   double truncationLimit)
    const
  {
    Cluster noise;
    std::vector<Cluster> tmp;

    RerunClustering(tmp, noise, epsilon, mPts, truncationLimit);

    return tmp.size();
  }


  double
  MergedCandidate::GetNExpectedPointsInCluster()
    const
  {
    const int n = GetNumberT2sCluster();
    return TMath::Gamma(n + 1)/(TMath::Gamma(n - 2)*6);
  }


  uint
  MergedCandidate::GetNPointsInCluster()
    const
  {
    uint n = 0;
    for (const auto& c : fCluster)
      n += c.fData.size();
    return n;
  }


  double
  MergedCandidate::GetPlaneFrontQuality()
    const
  {
    if (!fCluster.size())
      return -1;
    const double r = GetNPointsInCluster()/GetNExpectedPointsInCluster();
    if (!std::isnan(r))
      return r;
    else
      return -1;
  }


  double
  MergedCandidate::GetPhi()
    const
  {
    utl::Accumulator::WeightedMean phi;
    for (const auto& c : fCluster) {
      TVector3 axis(c.fu, c.fv, sqrt(1 - c.fu*c.fu - c.fv*c.fv));
      phi(axis.Phi(), c.fData.size());
    }
    return fCluster.size() ? phi.GetMean()*180/kPi : 0;
  }


  double
  MergedCandidate::GetTheta()
    const
  {
    utl::Accumulator::Mean cosTheta;
    for (const auto& c : fCluster) {
      for (const auto& p : c.fData)
        cosTheta(sqrt(1. - pow(double(p.fu), 2) - pow(double(p.fv), 2)));
    }
    return fCluster.size() ? 180./kPi*acos(cosTheta.GetMean()) : 0;
  }


  uint
  MergedCandidate::GetMaxSizeOfGraph()
    const
  {
    uint max = 0;
    for (const auto& g : fGraphs) {
      if (g.fNodesSignal.size() > max)
        max = g.fNodesSignal.size();
    }
    return max;
  }


  uint
  MergedCandidate::GetNumberOfT2s()
    const
  {
    return GetT2s().size();
  }


  std::vector<T2Data>
  MergedCandidate::GetT2s(const bool addNoise)
    const
  {
    std::vector<T2Data> tmp;
    for (const auto& g : fGraphs) {
      for (const auto& t2 : g.fNodesSignal)
        tmp.emplace_back(t2.fTime*kMicroSecond, t2.fId, t2.fTriggers);
    }
    for (const auto& c : fCluster) {
      for (const auto& t2 : c.fT2s) {
        if (std::find(tmp.begin(), tmp.end(), t2) == tmp.end()) {
          tmp.push_back(t2);
        }
      }
    }

    if (!addNoise)
      return tmp;
    for (const auto& gn : fNoiseT2s) {
      t2::T2Data t2(gn.fTime*kMicroSecond, gn.fId, gn.fTriggers);
      if (std::find(tmp.begin(), tmp.end(), t2) == tmp.end())
        tmp.push_back(t2);
    }

    return tmp;
  }


  std::vector<T2Data>
  MergedCandidate::GetT2s(const int id, const bool addNoise,
                          const bool convertToMeter)
    const
  {
    const auto conversionFactor = convertToMeter ? kMicroSecond : 1;

    std::vector<T2Data> tmp;
    for (const auto& g : fGraphs) {
      for (const auto& t2 : g.fNodesSignal) {
        if (t2.fId == id)
          tmp.emplace_back(t2.fTime * conversionFactor, t2.fId, t2.fTriggers);
      }
    }
    for (const auto& c : fCluster) {
      for (const auto& t2 : c.fT2s) {
        if (std::find(tmp.begin(), tmp.end(), t2) == tmp.end()) {
          if (t2.fId == id) {
            T2Data t2tmp(t2.fTime / kMicroSecond * conversionFactor,
                         t2.fId, t2.fTriggers);
            tmp.push_back(t2tmp);
          }
        }
      }
    }

    if (!addNoise)
      return tmp;
    for (const auto& gn : fNoiseT2s) {
      t2::T2Data t2(gn.fTime * conversionFactor, gn.fId, gn.fTriggers);
      if (t2.fId != id)
        continue;
      if (std::find(tmp.begin(), tmp.end(), t2) == tmp.end())
        tmp.push_back(t2);
    }

    return tmp;
  }

  std::vector<T2Data>
  MergedCandidate::GetCommonT2s()
    const
  {
    std::vector<T2Data> tmp;
    for (const auto& gn : fCommonT2s)
      tmp.emplace_back(gn.fTime * kMicroSecond, gn.fId, gn.fTriggers);
    return tmp;
  }


  std::vector<T2Data>
  MergedCandidate::GetGraphT2s()
    const
  {
    std::vector<T2Data> tmp;
    for (const auto& g : fGraphs) {
      for (const auto& t2 : g.fNodesSignal) {
        if (std::find(fCommonT2s.begin(), fCommonT2s.end(), t2) != fCommonT2s.end())
          continue;
        tmp.emplace_back(t2.fTime*kMicroSecond, t2.fId, t2.fTriggers);
      }
    }
    return tmp;
  }


  std::vector<T2Data>
  MergedCandidate::GetClusterT2s()
    const
  {
    std::vector<T2Data> tmp;
    const auto graphT2s = GetCommonT2s();
    for (const auto& c : fCluster) {
      for (const auto& t2 : c.fT2s) {
        if (std::find(graphT2s.begin(), graphT2s.end(), t2) != graphT2s.end())
          continue;
        tmp.push_back(t2);
      }
    }
    return tmp;
  }


  std::vector<T2Data>
  MergedCandidate::GetNoiseT2s()
    const
  {
    std::vector<T2Data> tmp;
    for (const auto& t2 : fNoiseT2s)
      tmp.emplace_back(t2.fTime*kMicroSecond, t2.fId, t2.fTriggers);
    return tmp;
  }


  std::vector<GraphNode>
  MergedCandidate::GetGraphNodes()
    const
  {
    std::vector<GraphNode> tmp;
    for (const auto& g : fGraphs) {
      for (const auto& t2 : g.fNodesSignal)
        tmp.emplace_back(t2);
    }
    std::sort(tmp.begin(), tmp.end());
    return tmp;
  }


  uint
  MergedCandidate::GetNumberOfDeadStations()
    const
  {
    std::vector<uint> deadIds;
    for (const auto& g : fGraphs) {
      for (const auto& n : g.fDeadNeighbours) {
        if (std::find(deadIds.begin(), deadIds.end(), n.fId) == deadIds.end())
          deadIds.push_back(n.fId);
      }
    }

    for (const auto& c : fCluster) {
      for (const auto& n : c.fDeadNeighbours) {
        if (std::find(deadIds.begin(), deadIds.end(), n.fId) == deadIds.end())
          deadIds.push_back(n.fId);
      }
    }
    return deadIds.size();
  }


  //meant to distinguish different types of
  // events (standard, inclined, SD-rings, ...)
  // might include additional parameters e.g. number of T2s
  int
  MergedCandidate::AssignEventClass(bool ignoreT3)
  {
    if (ContainsT3() && !ignoreT3) {
      if (fGraphs.size() == 1) {
        fType = EventType::T3;
      } else {
        int nT3 = 0;
        //loops only over compact formations, as they have to
        // contain all T3 configurations and are different from another
        for (const auto& g : fGraphs) {
          if (g.ContainsT3())
            ++nT3;
        }
        if (nT3 > 1)
          fType = EventType::MultiT3;
        else
          fType = EventType::T3;
      }
    } else if (!fCluster.size() && fGraphs.size()) {
      if (fGraphs.size() == 1)
        fType = EventType::SingleGraph;
      else
        fType = EventType::MultiGraph;
    } else if (!fGraphs.size() && fCluster.size()) {
      if (fCluster.size() == 1)
        fType = EventType::SingleCluster;
      else
        fType = EventType::MultiCluster;
    } else {
      fType = EventType::Combined;
    }

    return GetEventClass();
  }


  int
  MergedCandidate::GetEventClass()
    const
  {
    return int(fType);
  }


  /* This was changed at some point ... however I don't know when
    FIXME
    Key abcd: (for now still in decimal system to keep it human readable)
      a: type of event,
        0: T3
        1: cluster without graphs
        2: one graph without cluster
        3: > 1 graph without cluster
        4: graph(s) and cluster
      b: resevered for ToT-fraction classification,
         implemented as (+ 100 for more than 1 ToT), except for T3
      (b)cd: 'spectral' classification, a. k. a. number of
          ... t2s, clusterpoints, ...
          specific to each class (a)

    change to string output:
      - use same interface as pattern classification
      - can rely on fGraphs.front().GetMinKey(true) if desired
  */
  std::string
  MergedCandidate::GenerateKey()
    const
  {
    int nT2 = 0;
    std::string output = "";

    //compactness conditions for 3 T2 graph case
    int compactness = 0;
    int nToT = 0;

    std::pair<int, int> compactnessNtot;

    switch (fType) {
      case (EventType::Undefined):
        return std::to_string(-1);
      break;

      case (EventType::SingleCluster):
        output = "cluster-";
        //should occur only in very inclined events -> else combined event
        //separate small shower (2 stations) + X from 'X' alone
        if (fMinCrown == 1) {
          output += "nn-";
        }

        nT2 = GetNumberT2sCluster();
        output += std::to_string(nT2) + "-";
        if (GetToTFraction() > 1./nT2) {// fraction or absolute number?
          output += "tot";
        }

        if (GetPlaneFrontQuality() < 0.5)
          output += "-lq";
      break;

      case (EventType::SingleGraph):
        //return fGraphs.front().GetMinKey(true);
      {
        const auto& graph = fGraphs.front();
        compactnessNtot = graph.GetConfiguration();
        nT2 = GetNumberT2sGraphs();
        nToT = compactnessNtot.second;
        compactness = compactnessNtot.first;
        output = "graph-";

        //somewhat hierarchical trigger scheme:
        // the smaller the generated key the more random bg is expected
        if (nT2 > 5) {
          output += "big";
          if (graph.fAssociatedTriplets.size() > 2) {
            output += "-coherent";
          }
          //if (nT2 > 7)  //further selection, as still too much bg for 'singular' events
          output += "-" + std::to_string(nT2);
          break;
        }

        if (graph.fAssociatedTriplets.size() ||
            graph.fIsAligned) {
          if (graph.fAssociatedTriplets.size())
            output += "triplet";
          else if (graph.fIsAligned)
            output += "aligned";

          if (compactness > 2000)
            output += "-nn";
          if (compactness > 2300) {
            if (nToT == 2)
              output += "-2tot-2C1-3C2";
            else if (nToT == 1)
              output += "-1tot-2C1-3C2";
            else
              output += "-2C1-3C2";
          } else if (compactness > 1300) {
            if (GetToTFraction() > 0.01) {
              output += "-tot-3C2";
            } else {
              output += "-3C2";
            }
          } else if (compactness > 1130) {
            output += "-3C3";
          } else {
            output += std::to_string(graph.fNodesSignal.size());
          }

          if (graph.fDeadNeighbours.size())
            output += "-dead";
          break;
        }

        if (compactness > 2300) {
          output += "2C1-3C2";
          break;
        }

        if (nT2 > 2) {
          output += ">2T2";
        } else if (nToT == 2) {
          output += "nn-2tot";
        } else if (nToT) {
          output += "nn-1tot";
        } else {
          output += "nn";
        }
      }
      break;

      case (EventType::MultiGraph):
        output += "MultiGraph-";
        nT2 = GetNumberT2sGraphs();
        if (GetToTFraction() > 1./nT2)
          output += "tot-";
        if (fMultiGraphTriplets.size())   //'coherent'
          output += "coherent-";
        if (GetMaxSizeOfGraph() > 4)
          output += "big-";
        output += std::to_string(nT2) + "t2-";
        output += std::to_string(fGraphs.size()) + "graphs";
      break;

      case (EventType::MultiCluster):
        output = "MultiCluster-";
        output += std::to_string(GetNumberT2sCluster()) + "t2";
        if (GetPlaneFrontQuality() < 0.5)
          output += "-n";
      break;

      case (EventType::Combined):
        output = "Combined-";
        if (GetToTFraction() > 0.1) {
          output += "tot-";
        }
        if (fCommonClusterPoints.size() > 1)  // 'coherent'
          output += "coherent-";
        if (GetMaxSizeOfGraph() < 3) //essentially 1xxx events with a random doublet
          output += "nn-";
        output += std::to_string(GetNumberOfT2s()) + "t2";
        if (GetPlaneFrontQuality() < 0.5)
          output += "-n";
      break;

      case (EventType::T3):
        output = "T3-" + std::to_string(GetNumberT2sGraphs());
        if (ContainsCluster() && GetPlaneFrontQuality() < 0.5)
          output += "-n";
      break;

      case (EventType::MultiT3):
      {
        output = "MultiT3-";
        int nT3 = 0;
        for (const auto& g : fGraphs) {
          if (g.ContainsT3())
            ++nT3;
        }
        output += std::to_string(nT3) + "-T3s-";
        output += std::to_string(GetNumberOfT2s()) + "-t2";
        if (ContainsCluster() && GetPlaneFrontQuality() < 0.5)
          output += "-n";
      }
      break;

      default:
        std::cerr << "this should never happen! " << std::endl;
    }
    return output;
  }


  //loops only over graphs, as a the signature for XAS is defined
  // as cluster without graph
  std::vector<std::string>
  MergedCandidate::GenerateKeys()
    const
  {
    std::vector<std::string> tmp;
    for (const auto& g : fGraphs) {
      MergedCandidate m(g);
      m.AssignEventClass();
      tmp.push_back(m.GenerateKey());
    }

    return tmp;
  }


  //`high' level methods
  bool
  MergedCandidate::IsInteresting()
  {
    if (!fGPSSecond)
      return false;

    if (fCluster.size())
      return true;

    if (!fGraphs.size()) {
      std::cerr << "this should never happen (IsInteresting)!" << std::endl;
      return false;
    }

    if (fGraphs.size() > 2) {
      return true;
    } else if (fGraphs.size() == 2) {
      if (fMultiGraphTriplets.size())
        return true;
      return fGraphs[0].IsInteresting() || fGraphs[1].IsInteresting();
    }

    return fGraphs.front().IsInteresting();
  }


  bool
  MergedCandidate::FileOutput()
    const
  {
    switch (fType) {
      case (EventType::Undefined):
        return false;
      break;

      case (EventType::SingleCluster):
        return true;
      break;

      case (EventType::SingleGraph):
        return GenerateKey() > "2700"; // ?
      break;

      case (EventType::MultiGraph):
        return true;
      break;

      case (EventType::MultiCluster):
        return true;
      break;

      case (EventType::Combined):
        return true;  // ?
      break;

      case (EventType::T3):
        return true;
      break;

      case (EventType::MultiT3):
        return true;
      break;
    }
    return false;
  }


  //adds additional data, sets members
  void
  MergedCandidate::FillInformation(const std::vector<T2Data>& t2s,
                                   const std::vector<rTriplet>& triplets)
  {
    AddMultiGraphTriplets(triplets);
    AddNoise(t2s, triplets);
    FindCommonT2s();
    FindCommonClusterPoints();
    AssignEventClass();

    fToTFraction = GetToTFraction();
    fnT2s = GetNumberOfT2s();
    fPlaneQuality = GetPlaneFrontQuality();
  }
};
