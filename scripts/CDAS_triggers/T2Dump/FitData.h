#include <TVector3.h>
#include <T2Dump/T2Triplet.h>

#ifndef _FitData_
#define _FitData_

namespace FitData
{
  extern int fTimes[3];
  extern TVector3 fdeltaX[3];

  double
  GetTHatZero(const double& phi, const double& t0, const TVector3& deltaX);

  void
  SetFitData(const T2Triplet<TVector3>& t);

  /*
  parameter:
   par[0] is t0
   par[1] is phi (rad)
  */
  void
  fcnGausThetaPhi(int &npar, double *gin, double &f, double *par, int iflag);

}

#endif