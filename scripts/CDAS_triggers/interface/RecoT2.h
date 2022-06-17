#ifndef _recoT2_
#define _recoT2_

#include <Rtypes.h>
#include <iostream>
#include <utl/Accumulator.h>
#include <TH1F.h>

//saves the reconstructed triplet, with time and direction
struct ReconstructedT2
{
  unsigned int fGPSSecond = 0;
  int fMicroSecond = 0;

  float fu = 0;
  float fv = 0;

  float fDistance[3];
  ushort fIds[3];
  ushort fAdditionalMatches = 0;

  ReconstructedT2(unsigned int GPSSecond) : fGPSSecond(GPSSecond) {}
  ReconstructedT2(unsigned int GPSSecond, int microSecond) 
    :  fGPSSecond(GPSSecond), fMicroSecond(microSecond) {}
  ReconstructedT2() {}
  ~ReconstructedT2() {}

  bool ContainsId(int id) const { return fIds[0] == id || fIds[1] == id || fIds[2] == id; }

  double GetTheta() const { return acos(sqrt(1 - fu*fu - fv*fv)); }
  double GetPhi() const { return atan(fv/fu); }
  double GetAvgDistance() const { return 1./3*(fDistance[0] + fDistance[1] + fDistance[2]); }

  ClassDefNV(ReconstructedT2, 3);
};

#endif
