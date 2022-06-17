#ifndef _Conditions_h_
#define _Conditions_h_

#include <vector>
#include <iostream>
#include <sstream>
#include <string>
#include <utl/String.h>
#include <exception>
#include <interface/MergedCandidate.h>
#include <interface/CandidateCondition.h>
#include <boost/scoped_ptr.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/exceptions.hpp>
#include <Rtypes.h>


namespace bpt = boost::property_tree;

namespace t2 {


  struct OutputConditions {
    std::vector<CandidateCondition> fConditions;

    explicit operator bool() const
    { return fConditions.size(); }

    void
    ReadBpt(const bpt::ptree& t)
    {
      for (const auto& condition : t.get_child("OutputConditions")) {
        fConditions.emplace_back();
        auto& c = fConditions.back();
        c.ConvertNameToType(condition.first);
        c.ReadBpt(condition.second);
      }
    }

    bool
    ApplyConditions(const MergedCandidate& m, const bool verbose = false)
      const
    {
      for (const auto& c : fConditions) {
        if (c.ApplyCondition(m, verbose)) {
          return true;
        }
      }
      return false;
    }

    void
    ReadFile(const std::string& filename)
    {
      std::cout << "reading config file: " << filename << std::endl;
      std::ifstream in(filename);
      std::string line;
      while (std::getline(in, line)) {
        if (line.find("#") != std::string::npos) { //ignore line with #
          continue;
        }

        //check first 'word'
        std::stringstream tmp(line);
        std::string type;
        tmp >> type;

        //check `special' commands to define conditions
        bool newCondition = true;
        if (type == "SingleCluster") {
          fConditions.emplace_back(EventType::SingleCluster);
        } else if (type == "SingleGraph") {
          fConditions.emplace_back(EventType::SingleGraph);
        } else if (type == "MultiGraph") {
          fConditions.emplace_back(EventType::MultiGraph);
        } else if (type == "MultiCluster") {
          fConditions.emplace_back(EventType::MultiCluster);
        } else if (type == "Combined") {
          fConditions.emplace_back(EventType::Combined);
        } else if (type == "T3") {
          fConditions.emplace_back(EventType::T3);
        } else if (type == "MultiT3") {
          fConditions.emplace_back(EventType::MultiT3);
        } else {
          newCondition = false;
        }
        if (newCondition)
          continue;

        if (fConditions.size())
          fConditions.back().FillParameter(line);
      }
      std::cout << "found " << fConditions.size()
                << " conditions." << std::endl;
    }
  };


  inline
  std::ostream&
  operator<<(std::ostream& os, const OutputConditions& c)
  {
    os << c.fConditions.size() <<  " Outputconditions: "
       << std::endl;
    for (const auto& cond : c.fConditions) {
      os << " " << cond << std::endl;
    }
    return os;
  }

};
#endif