#ifndef _Binning_h
#define _Binning_h

#include <interface/CandidateCondition.h>
#include <interface/MergedCandidate.h>
#include <T2Dump/Triplet.h>
#include <boost/scoped_ptr.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/exceptions.hpp>
#include <Rtypes.h>
#include <vector>


namespace bpt = boost::property_tree;

namespace t2 {

  struct TripletCondition {
    bool fContainsNN = true;  // (fContainsNN || triplet.nn())
    
    uint fnMinToT = 0;
    uint fnMaxToT = 4;

    double fMinAvgDistance = 0;
    double fMaxAvgDistance = 100000;

    uint fnPassingCondition = 0;

    void
    ReadBpt(const bpt::ptree& t)
    {
      try {
        fContainsNN = t.get<bool>("nn", true);
        
        fnMinToT = t.get<int>("nMinToT", 0);
        fnMaxToT = t.get<int>("nMaxToT", 4);
        fMinAvgDistance = t.get<double>("minDistance", 0);
        fMaxAvgDistance = t.get<double>("maxDistance", 1e5);
      } catch (const std::exception& e) {
        std::cerr << "error filling values" << e.what()
                  << std::endl;
      }
    }

    bool
    ApplyCondition(const rTriplet& t)
      const
    {
      return (fContainsNN || t.ContainsNN()) && 
             fnMinToT <= t.GetNToT() && 
             t.GetNToT() <= fnMaxToT &&
             fMinAvgDistance <= t.GetAvgDistance() &&
             fMaxAvgDistance >= t.GetAvgDistance();
    }

    ClassDefNV(TripletCondition, 1);
  };


  inline
  std::ostream&
  operator<<(std::ostream& os, const TripletCondition& c)
  {
    os << "Triplet-Condition: "
       << c.fContainsNN << " (nn), "
       << c.fnMinToT << " <= nToT <=" << c.fnMaxToT << ", "
       << c.fMinAvgDistance << " <= avgDist <= " << c.fMaxAvgDistance;

    return os;
  }


  struct BinningConditions {
    uint fGPSSecond = 0;  //begin
    int fBinLength = 1/*s*/; 

    std::vector<CandidateCondition> fCandidateConditions;
    std::vector<TripletCondition> fTripletConditions;

    uint fnActiveStations = 0;

    void
    ReadBpt(const bpt::ptree& t)
    {
      for (const auto& condition : t.get_child("BinningConditions")) {
        if (condition.first == "triplet") {
          TripletCondition tmp;
          tmp.ReadBpt(condition.second);
          fTripletConditions.push_back(tmp);
        } else {
          CandidateCondition c;
          c.ConvertNameToType(condition.first);  
          c.ReadBpt(condition.second);
          fCandidateConditions.push_back(c);
        }
      }
    }

    void 
    AddData(const std::vector<rTriplet>& triplets)
    {
      if (fTripletConditions.empty())
        return;

      for (const auto& t : triplets) {
        if (t.fGPSSecond != fGPSSecond)
          continue;
        for (auto& p : fTripletConditions) {
          if (p.ApplyCondition(t))
            ++p.fnPassingCondition;
        }
      }        
    }

    bool
    HasPattern()
      const
    {
      for (const auto& p : fCandidateConditions) {
        if (p.fAllowedPatterns.size())
          return true;
      }
      return false;
    }

    //meant specifically for pattern searches based on graphs
    // i.e. before merging
    void
    AddData(const std::vector<Graph>& graphs)
    {
      if (!HasPattern())
        return;

      for (const auto& g : graphs) {
        MergedCandidate m(g);
        m.AssignEventClass();
        for (auto& p : fCandidateConditions) {
          if (p.ApplyCondition(m))
            ++p.fnPassingCondition;
        }
      }
    }

    void
    AddData(const std::vector<MergedCandidate>& candidates)
    {
      for (const auto& m : candidates) {
        for (auto& p : fCandidateConditions) {
          if (p.ApplyCondition(m))
            ++p.fnPassingCondition;
        }
      }
    }

    void
    ResetCounter()
    {
      for (auto& p : fCandidateConditions)
        p.fnPassingCondition = 0;
      for (auto& p : fTripletConditions)
        p.fnPassingCondition = 0;
    }

    BinningConditions() = default;
    ~BinningConditions() = default;

    ClassDefNV(BinningConditions, 2);
  };


  inline
  std::ostream&
  operator<<(std::ostream& os, const BinningConditions& c)
  {
    os << "Binning (candidate-level): " << std::endl;
    for (const auto& p : c.fCandidateConditions) {
      os << "  " << p << ": " << p.fnPassingCondition << std::endl;
    }
    if (c.fTripletConditions.size())
      os << "Binning (triplets): " << std::endl;
    for (const auto& p : c.fTripletConditions) {
      os << "  " << p << ": " << p.fnPassingCondition << std::endl;
    }
    return os;
  }
};
#endif
