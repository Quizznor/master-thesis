#ifndef _projPos_
#define _projPos_

#include <vector>
#include <TVector3.h>
#include <utility>
#include <string>
#include <healpix_base.h>
#include <pointing.h>

class ProjectedPositions
{
private:
  //keeps axis*(deltaPosition) in mu s for different directions
  //binning is: bin i in cos theta and bin j in Phi:
  //    -> binnumber = i*(fnBinsPhi) + j  for bins starting with 0
  std::vector<double> fProjectedTimeDifferences[2000];
  TVector3 fRealPositions[2000];    // - reference point is applied

  std::vector<double> fSigmaTrefEstimates[2000];

  unsigned fnBinsPhi = 60;
  unsigned fnBinsCosTheta = 30;

  T_Healpix_Base<int> fHealPixBase;

  std::string fFilename = "/home/schimassek/SVN/T2Scalers/ms/src/Data/SdPositions.txt";
  TVector3 fReferencePoint = TVector3(473872./300., 6104846./300., 1500./300.);

public:
  ProjectedPositions();
  ProjectedPositions(std::string filename, const TVector3& ref);
  ~ProjectedPositions() {}

  double GetRealDistance(uint id1, uint id2) const;
  TVector3 GetDeltaX(uint id1, uint id2) const
   { return fRealPositions[id1] - fRealPositions[id2];}

  //void SetNBins(unsigned nCosTheta, unsigned nPhi);
  void ReadPositions(std::string file, const TVector3& referencePoint);
  //int GetBin(double cosTheta, double phi) const;
  std::pair<double, double> GetBinCenter(unsigned binNumber) const;
  //int GetNBinsCosTheta() const { return fnBinsCosTheta; }
  //int GetNBinsPhi() const { return fnBinsPhi; }
  uint GetNBins() const { return fHealPixBase.Npix()/2 + fHealPixBase.Nside()*2; }
  bool IdExists(int id) const { return bool(fProjectedTimeDifferences[id].size() > 0); }

  double GetBinWidth(int bin = 1) const;
  void SetOrder(int order);

  //std::vector<unsigned> GetNeighbouringBins(unsigned bin);
  double GetVarAX(unsigned bin, const TVector3& deltaX) const;
  double GetVarAXEstimate(unsigned bin, unsigned id) const;
  double GetSigmaTref(unsigned bin, unsigned id) const;

  std::vector<double>& operator[](unsigned id) 
    { return fProjectedTimeDifferences[id]; }

  TVector3 GetPosition(const unsigned id) const
  { return fRealPositions[id]; }

  friend class IDataHandler;
};

#endif

