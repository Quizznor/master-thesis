#ifndef _TimeStamp_
#define _TimeStamp_

#include <Rtypes.h>
#include <vector>
#include <utility>

class ProjectedPositions;

struct TimeStamp
{
  unsigned int fGPSSecond = 0;
  double fMicroSecond = 0;

  std::vector<ushort> fIds;
  std::vector<std::pair<float, float> > fReconstructedTrefs;  //(value, sigma)

  unsigned short fDirectionBin = 0;
  unsigned short fAvgDistance = 0;
  float fchiSquare = 0;
  float fTimeSpread = 0;

  double GetAvgDistance(const ProjectedPositions&) const;
  double GetChiSquare() const;
  std::vector<double> GetCorrelationMatrixEstimate(const ProjectedPositions&) const;
  TimeStamp RemoveAccidentals(double threshold = 5);
  void erase(unsigned first, unsigned last);  

  unsigned size() const { return fIds.size(); }
  void reset() { *this = TimeStamp(); }
  ClassDefNV(TimeStamp, 5);
};

#endif