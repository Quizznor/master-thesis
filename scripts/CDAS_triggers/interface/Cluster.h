#ifndef _Cluster_
#define _Cluster_

#include <Rtypes.h>

struct Cluster
{
  std::vector<ReconstructedT2> fMemberData;

  double fu = 0;
  double fv = 0;

  double fVarU = 0;
  double fVarV = 0;

  Cluster() {}
  ~Cluster() {}

  Cluster(const std::vector<ReconstructedT2>& r, 
          const std::vector<int> labels, 
          int label)
  {
    utl::Accumulator::Var MeanVarU;
    utl::Accumulator::Var MeanVarV;
    if (r.size() == labels.size()) {
      for (uint i = 0; i < r.size(); ++i) {
        if (labels[i] == label) {
          fMemberData.push_back(r[i]);
          MeanVarU(r[i].fu);
          MeanVarV(r[i].fv);
        }
      }
      fu = MeanVarU.GetMean();
      fv = MeanVarV.GetMean();
      if (fMemberData.size() > 2) {
        fVarU = MeanVarU.GetVar();
        fVarV = MeanVarV.GetVar();
      }
    }    
  }

  void
  GetAdditionalMatchesStats(double& mean, double& var) const
  {
    utl::Accumulator::Var m;
    for (const auto& r : fMemberData)
      m(r.fAdditionalMatches);

    mean = m.GetMean();
    var = m.GetVar();
  }

  ClassDefNV(Cluster, 2);
};

#endif