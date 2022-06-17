#ifndef _JoinedWithCDAS_h_
#define _JoinedWithCDAS_h_

#include <interface/MergedCandidate.h>

namespace t2 {

  struct CandidateJoinedWithCDAS : public MergedCandidate {
    t3::T3Data fT3;

    bool fContainsDataLoss = false;

    CandidateJoinedWithCDAS() = default;
    CandidateJoinedWithCDAS(const MergedCandidate& m) :
      MergedCandidate::MergedCandidate(m) {}
    CandidateJoinedWithCDAS(const MergedCandidate& m,
       const t3::T3Data& t3) :
      MergedCandidate::MergedCandidate(m), fT3(t3) {}

    bool HasLostStationData() const;

    ClassDefNV(CandidateJoinedWithCDAS, 2);
  };

  inline
  std::ostream&
  operator<<(std::ostream& os, const CandidateJoinedWithCDAS& m)
  {
    os << "Joined[ " << m.fT3 << " | ";
    os << m.fGPSSecond
       << ", " << m.fMicroSecond
       << ", " << m.GetEventClass()
       << ", " << m.fGraphs.size();
    if (m.fType == EventType::T3) {
      for (const auto& n : m.fGraphs.front().fNodesSignal)
        os << ", " << n;
    }
    os << "]";
    return os;
  }
};
#endif  