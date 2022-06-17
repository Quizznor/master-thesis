#ifndef _fittedT2_
#define _fittedT2_

#include <Rtypes.h>

struct FittedT2
{
  const uint fGPSSecond = 0;
  float fTheta = 0;
  float fPhi = 0;
  float fT0 = 0;
  double fFcn = 0;
  float fTestStat = 3;
  float fD = 0;

  float fThetaErr = 0;
  float fPhiErr = 0;
  float fT0Err = 0;
  double fAverageDistance = 0;

  FittedT2() {}
  ~FittedT2() {}
  FittedT2(uint gps) : fGPSSecond(gps) {}

  ClassDefNV(FittedT2, 1);
};

#endif