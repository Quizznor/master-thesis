#include <T2Dump/FitData.h>
#include <iostream>

namespace FitData{
  int fTimes[3];
  TVector3 fdeltaX[3];

  double
  GetTHatZero(double phi, 
              double theta,
              double t0, 
              const TVector3& deltaX)
  {
    TVector3 axis;
    axis.SetX(1.);
    axis.SetPhi(phi);
    axis.SetTheta(theta);

    return -axis*deltaX + t0;
  }

  void
  SetFitData(const T2Triplet<TVector3>& t)
  {
    for (int i = 0; i < 3; ++i)
      fTimes[i] = t.fTimes[i];

    fdeltaX[1] = t.fDeltaX[1];
    fdeltaX[2] = t.fDeltaX[2];
  }

  void
  fcnGausThetaPhi(int &npar, double */*gin*/, double &f, double *par, int /*iflag*/)
  { 
    fdeltaX[0] = TVector3();
    if (npar > 3)
      return;
    f = 0;

    f += sqr(FitData::fTimes[0] - GetTHatZero(par[2], par[1], par[0], FitData::fdeltaX[0]));
    f += sqr(FitData::fTimes[1] - GetTHatZero(par[2], par[1], par[0], FitData::fdeltaX[1])); 
    f += sqr(FitData::fTimes[2] - GetTHatZero(par[2], par[1], par[0], FitData::fdeltaX[2]));
  }
};