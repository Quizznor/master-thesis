#ifndef _CandConditions_h_
#define _CandConditions_h_

#include <vector>
#include <iostream>
#include <sstream>
#include <string>
#include <utl/String.h>
#include <exception>
#include <interface/MergedCandidate.h>
#include <boost/scoped_ptr.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/exceptions.hpp>
#include <Rtypes.h>


namespace bpt = boost::property_tree;

namespace t2 {

  struct CandidateCondition {
    EventType fType = EventType::Undefined;

    uint fMinGPS = 0;
    uint fMaxGPS = 2000000000;

    uint fMinNumberOfT2s = 0;
    int fMaxNumberOfT2s = -1;
    uint fMinNumberOfT2sCluster = 0;
    uint fMinNumberOfT2sGraph = 0;

    uint fMinNumberOfClusters = 0;
    uint fMinNumberOfGraphs = 0;
    uint fMinNumberOfCommonClusterPoints = 0;
    uint fMinNumberOfCommonT2s = 0;

    float fMinToTFraction = 0;
    uint fMinNumberOfTriplets = 0;
    uint fMaxNumberDeadNeighbours = 2000;

    int fMinNumberDoubleTrigger = 0;
    int fMaxNumberDoubleTrigger = -1;

    double fMinPointSourceSpread = 0;
    double fMaxPointSourceSpread = -1;

    int fMinNWide = -3; //> HACK-ALERT: I indicate events before the
                        //  introduction of the new flags with -2. Old events
                        //  are initialised with -1
    int fMaxNWide = 0;

    double fMinNWideRatio = -3; //> NOTE chosen such that default values do not remove any candidate
    double fMaxNWideRatio = 2;

    //Graphs
    bool fAligned = false;  //needs to be aligned
    std::vector<std::string> fAllowedPatterns;  //e.g. 3-1-(0,1,1)-...
    uint fMinCompactness = 0;
    uint fMinMultiGraphTriplets = 0;

    //Cluster
    float fMinPlaneQuality = -2;  //-1 is for no plane ...

    //Counter
    uint fnPassingCondition = 0;

    void
    FillParameter(const std::string& line)
    {
      std::cerr << "WARNING: this method is not maintained for newer members!"
                   " [from CandidateCondition::FillParameter(const std::string)]"
                << std::endl;

      std::stringstream ss(line, std::ios_base::in);
      std::string key;
      if (line.empty())
        return;
      try {
        if (!(ss >> key))
          std::cerr << "error converting line" << std::endl;
      } catch(const std::exception& e) {
        std::cerr << "error converting line: "
                  << e.what()
                  << std::endl;
      }
      try {
        if (key == "nMinT2") {
          ss >> fMinNumberOfT2s;
        } else if (key == "nMinT2Cluster") {
          ss >> fMinNumberOfT2sCluster;
        } else if (key == "nMinT2Graph") {
          ss >> fMinNumberOfT2sGraph;
        } else if (key == "nMinCluster") {
          ss >> fMinNumberOfClusters;
        } else if (key == "nMinGraph") {
          ss >> fMinNumberOfGraphs;
        } else if (key == "nMinCommonClusterPoints") {
          ss >> fMinNumberOfCommonClusterPoints;
        } else if (key == "nMinCommonT2") {
          ss >> fMinNumberOfCommonT2s;
        } else if (key == "nMinToTFraction") {
          ss >> fMinToTFraction;
        } else if (key == "nMinTriplets") {
          ss >> fMinNumberOfTriplets;
        } else if (key == "nMaxDead") {
          ss >> fMaxNumberDeadNeighbours;
        } else if (key == "aligned") {
          ss >> fAligned;
        } else if (key == "pattern") {
          std::string tmp;
          while (ss >> tmp >> std::ws)
            fAllowedPatterns.push_back(tmp);
        } else if (key == "minCompactness") {
          ss >> fMinCompactness;
        } else if (key == "nMinMultiGraphTriplet") {
          ss >> fMinMultiGraphTriplets;
        } else if (key == "minQuality") {
          ss >> fMinPlaneQuality;
        } else {
          std::cerr << "unknown key: " + key << std::endl;
        }
      } catch (std::exception& e) {
        std::cerr << "error reading value " << e.what()
                  << std::endl;
      }
    }

    bool
    CheckAligned(const t2::MergedCandidate& m)
      const
    {
      if (!m.fGraphs.size())
        return false;
      for (const auto& g : m.fGraphs) {
        if (g.fIsAligned)
          return true;
      }
      return false;
    }

    bool
    CheckPattern(const t2::MergedCandidate& m)
      const
    {
      if (!fAllowedPatterns.size())
        return true;

      for (const auto& g : m.fGraphs) {
        const auto pattern = g.GetMinKey(true);
        if (std::find(fAllowedPatterns.begin(), fAllowedPatterns.end(),
              pattern) != fAllowedPatterns.end()) {
          return true;
        }
      }
      return false;
    }

    bool
    CheckCompactness(const MergedCandidate& m, const bool verbose = false)
      const
    {
      if (!fMinCompactness)
        return true;

      for (const auto& g : m.fGraphs) {
        const auto compactness = g.fCompactness < 0 ?
          g.GetConfiguration().first : g.fCompactness;
        if (verbose)
          std::cout << "compactness: " << compactness
                    << " >= " << fMinCompactness
                    << " ";
        if (compactness >= int(fMinCompactness))
          return true;
      }
      return false;
    }

    bool
    ApplyCondition(const t2::MergedCandidate& m, const bool verbose = false)
      const
    {
      if (fType != m.fType)
        return false;

      const int nWide = m.GetNWideTriggers();
      const double wideRatio = double(nWide) / m.fnT2s;

      if (verbose) {
        std::cout << "applying: " << fMinGPS << " < " << m.fGPSSecond << " && "
                  << fMaxGPS << " >= " << m.fGPSSecond << " && "
                  << fMinNumberOfT2s << " <= " << m.GetNumberOfT2s() << " && "
                  << "(" << fMaxNumberOfT2s << "< 1 || " << int(m.GetNumberOfT2s())
                  << " <= " << fMaxNumberOfT2s << " ) && "
                  << fMinNumberOfClusters << " <= " << m.fCluster.size() << " && "
                  << fMinNumberOfGraphs << " <= " << m.fGraphs.size() << " && "
                  << fMinNumberOfT2sCluster << " <= " << m.GetNumberT2sCluster() << " && "
                  << fMinNumberOfT2sGraph << " <= " << m.GetNumberT2sGraphs() << " && "
                  << fMinNumberOfCommonClusterPoints << " <= " <<
                        m.fCommonClusterPoints.size() << " && "
                  << fMinNumberOfCommonT2s << " <= " << m.fCommonT2s.size() << " && "
                  << fMinToTFraction << " <= " << m.GetToTFraction() << " && "
                  << fMinNumberOfTriplets << " <= " << m.GetNumberTriplets() << " && "
                  << fMaxNumberDeadNeighbours << " >= " << m.GetNumberOfDeadStations() << " && "
                  << "CheckPattern(m) " << CheckPattern(m) << " && "
                  << (!fAligned || CheckAligned(m)) << " && "
                  << CheckCompactness(m, true) << " [compactness] && "
                  << fMinMultiGraphTriplets << " <= " << m.fMultiGraphTriplets.size() << " && "
                  << fMinPlaneQuality << " <= " << m.GetPlaneFrontQuality() << " && "
                  << (m.fPointSourceSpread < 0 || fMaxPointSourceSpread < 0
                      || m.fPointSourceSpread <= fMaxPointSourceSpread) << " fPointSourceSpread && "
                  << (m.fPointSourceSpread < 0 || m.fPointSourceSpread >= fMinPointSourceSpread) << " pointsource 2 && "
                  << (m.fNDoubleTrigger < 0 || fMinNumberDoubleTrigger <= m.fNDoubleTrigger) << " [double trigger] && "
                  << (m.fNDoubleTrigger < 0 || fMaxNumberDoubleTrigger < 0
                      || m.fNDoubleTrigger <= fMaxNumberDoubleTrigger) << " double 2 && "
                  << nWide << " > " << fMinNWide << " && "
                  << (!fMaxNWide || nWide <= fMaxNWide) << " wide 2 && "
                  << (wideRatio > fMinNWideRatio) << " wideratio && "
                  << (wideRatio < fMaxNWideRatio) << " wideratio 2\n";
      }

      return     fMinGPS <= m.fGPSSecond
              && fMaxGPS >= m.fGPSSecond
              && fMinNumberOfT2s <= m.GetNumberOfT2s()
              && (fMaxNumberOfT2s < 1 || int(m.GetNumberOfT2s()) <= fMaxNumberOfT2s)
              && fMinNumberOfClusters <= m.fCluster.size()
              && fMinNumberOfGraphs <= m.fGraphs.size()
              && fMinNumberOfT2sCluster <= m.GetNumberT2sCluster()
              && fMinNumberOfT2sGraph <= m.GetNumberT2sGraphs()
              && fMinNumberOfCommonClusterPoints <=
                    m.fCommonClusterPoints.size()
              && fMinNumberOfCommonT2s <= m.fCommonT2s.size()
              && fMinToTFraction <= m.GetToTFraction()
              && fMinNumberOfTriplets <= m.GetNumberTriplets()
              && fMaxNumberDeadNeighbours >= m.GetNumberOfDeadStations()
              && CheckPattern(m)
              && (!fAligned || CheckAligned(m))
              && CheckCompactness(m)
              && fMinMultiGraphTriplets <= m.fMultiGraphTriplets.size()
              && fMinPlaneQuality <= m.GetPlaneFrontQuality()
              && (m.fPointSourceSpread < 0 || fMaxPointSourceSpread < 0
                  || m.fPointSourceSpread <= fMaxPointSourceSpread)
              && (m.fPointSourceSpread < 0 || m.fPointSourceSpread >= fMinPointSourceSpread)
              && (m.fNDoubleTrigger < 0 || fMinNumberDoubleTrigger <= m.fNDoubleTrigger)
              && (m.fNDoubleTrigger < 0 || fMaxNumberDoubleTrigger < 0
                  || m.fNDoubleTrigger <= fMaxNumberDoubleTrigger)
              && nWide >= fMinNWide
              && (!fMaxNWide || nWide <= fMaxNWide)
              && wideRatio > fMinNWideRatio
              && wideRatio < fMaxNWideRatio;
    }

    void
    ReadBpt(const bpt::ptree& t)
    {
      try {
        fMinGPS = t.get<uint>("minGPS", 0);
        fMaxGPS = t.get<uint>("maxGPS", 2000000000);
        fMinNumberOfT2s = t.get<uint>("nMinT2", 0);
        fMaxNumberOfT2s = t.get<int>("nMaxT2", -1);
        fMinNumberOfT2sCluster = t.get<uint>("nMinT2Cluster", 0);
        fMinNumberOfT2sGraph = t.get<uint>("nMinT2Graph", 0);
        fMinNumberOfClusters = t.get<uint>("nMinCluster", 0);
        fMinNumberOfGraphs = t.get<uint>("nMinGraph", 0);
        fMinNumberOfCommonClusterPoints = t.get<uint>("nMinCommonClusterPoints", 0);
        fMinNumberOfCommonT2s = t.get<uint>("nMinCommonT2", 0);
        fMinToTFraction = t.get<float>("minToTFraction", 0.);
        fMinNumberOfTriplets = t.get<uint>("nMinTriplets", 0);
        fMaxNumberDeadNeighbours = t.get<uint>("nMaxDead", 2000);
        fAligned = t.get<bool>("aligned", false);

        fMinNumberDoubleTrigger = t.get<int>("nMinDoubleTrigger", 0);
        fMaxNumberDoubleTrigger = t.get<int>("nMaxDoubleTrigger", -1);

        fMinPointSourceSpread = t.get<double>("minPointSourceSpread", 0);
        fMaxPointSourceSpread = t.get<double>("maxPointSourceSpread", -1);

        fMinNWide = t.get<int>("minWide", -3);
        fMaxNWide = t.get<int>("maxWide", 0);

        fMinNWideRatio = t.get<double>("minWideRatio", -3);
        fMaxNWideRatio = t.get<double>("maxWideRatio", 2);

        try {
          fAllowedPatterns = utl::AsVectorOf<std::string>(
                                    t.get<std::string>("pattern"));
        } catch (const bpt::ptree_bad_path& e) {
          fAllowedPatterns.resize(0);
        }
        fMinCompactness = t.get<uint>("minCompactness", 0);
        fMinMultiGraphTriplets = t.get<uint>("nMinMultiGraphTriplet", 0);
        fMinPlaneQuality = t.get<float>("minQuality", -2.);
      } catch (const std::exception& e) {
        std::cerr << "error filling values" << e.what()
                  << std::endl;
      }
    }

    void
    ConvertNameToType(const std::string& name)
    {
      std::string type;
      std::istringstream ss(name);
      ss >> std::ws >> type;

      if (type == "SingleCluster") {
        fType = EventType::SingleCluster;
      } else if (type == "SingleGraph") {
        fType = EventType::SingleGraph;
      } else if (type == "MultiGraph") {
        fType = EventType::MultiGraph;
      } else if (type == "MultiCluster") {
        fType = EventType::MultiCluster;
      } else if (type == "Combined") {
        fType = EventType::Combined;
      } else if (type == "T3") {
        fType = EventType::T3;
      } else if (type == "MultiT3") {
        fType = EventType::MultiT3;
      } else {
        std::cerr << "unknown type: " << type << std::endl;
      }
    }

    CandidateCondition(const EventType& t) : fType(t) { }
    CandidateCondition() : fType(EventType::Undefined) { }
    ~CandidateCondition() = default;

    ClassDefNV(CandidateCondition, 1);
  };


  inline
  std::ostream&
  operator<<(std::ostream& os, const CandidateCondition& c)
  {
    os << "Type: ";
    switch (int(c.fType)) {
      case 0:
        os << "Undefined";
      break;

      case 1:
        os << "SingleCluster";
      break;

      case 2:
        os << "SingleGraph";
      break;

      case 3:
        os << "MultiGraph";
      break;

      case 4:
        os << "MultiCluster";
      break;

      case 5:
        os << "Combined";
      break;

      case 6:
        os << "T3";
      break;

      case 7:
        os << "MultiT3";
      break;

      default:
        os << " unknown";
    }
    const auto cDefault = CandidateCondition();

    if (c.fMinNumberOfT2s != cDefault.fMinNumberOfT2s)
      os << "; nT2: " << c.fMinNumberOfT2s;
    if (c.fMinNumberOfT2sCluster != cDefault.fMinNumberOfT2sCluster)
      os << "; nT2(Cluster): " << c.fMinNumberOfT2sCluster;
    if (c.fMinNumberOfT2sGraph != cDefault.fMinNumberOfT2sGraph)
      os << "; nT2(Graph): " << c.fMinNumberOfT2sGraph;
    if (c.fMinNumberOfClusters != cDefault.fMinNumberOfClusters)
      os << "; n(Cluster): " << c.fMinNumberOfClusters;
    if (c.fMinNumberOfGraphs != cDefault.fMinNumberOfGraphs)
      os << "; n(Graph): " << c.fMinNumberOfGraphs;
    if (c.fMinNumberOfCommonClusterPoints !=
        cDefault.fMinNumberOfCommonClusterPoints)
      os << "; n(C.cl.Points): " << c.fMinNumberOfCommonClusterPoints;
    if (c.fMinNumberOfCommonT2s != cDefault.fMinNumberOfCommonT2s)
      os << "; n(C.T2s): " << c.fMinNumberOfCommonT2s;
    if (c.fMinToTFraction != cDefault.fMinToTFraction)
      os << "; tot-fraction: " << c.fMinToTFraction;
    if (c.fMinNumberOfTriplets != cDefault.fMinNumberOfTriplets)
      os << "; n(Triplets): " << c.fMinNumberOfTriplets;
    if (c.fMaxNumberDeadNeighbours != cDefault.fMaxNumberDeadNeighbours)
      os << "; n(Dead): " << c.fMaxNumberDeadNeighbours;
    if (c.fAligned != cDefault.fAligned)
      os << "; aligned: " << c.fAligned;
    if (c.fAllowedPatterns.size())
      os << "; n(Pattern): " << c.fAllowedPatterns.size();
    if (c.fMinCompactness != cDefault.fMinCompactness)
      os << "; minCompactness: " << c.fMinCompactness;
    if (c.fMinMultiGraphTriplets != cDefault.fMinMultiGraphTriplets)
      os << "; n(MultiGraphT.): " << c.fMinMultiGraphTriplets;
    if (c.fMinPlaneQuality != cDefault.fMinPlaneQuality)
      os << "; qMin: " << c.fMinPlaneQuality;
    return os;
  }

};
#endif
