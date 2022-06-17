#ifndef _T2Triplet_h_
#define _T2Triplet_h_

#include <T2Dump/DataHandler.h>
#include <TVector3.h>
#include <iostream>
#include <exception>
#include <cmath>
#include <utility>
#include <algorithm>
#include <interface/FittedT2.h>
#include <interface/T2.h>
#include <interface/RecoT2.h>


template<typename T>
constexpr T sqr(const T t) { return t*t; }

template<class vec = TVector3>
struct T2Triplet {

  const int fMicroSecond = 0;   //can be also in meters, as long as distances are in meters as well
  uint fTimes[2] = { 0 };       //stores time differences to first T2 = fMicrosecond
  vec fDeltaX[2];
  uint fIds[3] = { 0 };

  double fTau1 = 1.5;
  double fTau2 = 1.5;
  double fij = 2;
  double falpha = 0;
  double fbeta = 0;
  double fD = 0;

  T2Triplet() { }
  template<class t2>
  T2Triplet(const t2& t) : fMicroSecond(t.fTime), fIds{t.fId, 0, 0} { }
  T2Triplet(int microSecond) : fMicroSecond(microSecond) {}


  //add a t2 to an incomplete triplet
  template<class data>
  bool
  AddT2(const data& t2, const vec& deltaX)
  {
    if (ContainsId(t2.fId))         //reject same station with later t2 with same id
      return false;

    if (!t2.fId)
      return false;

    if (fIds[1] == 0) {             //only the first T2 is set
      //check compatibility with speed of light
      if (deltaX.Mag2() < sqr(t2.fTime - fMicroSecond))
        return false;

      fTimes[0] = t2.fTime - fMicroSecond;
      fDeltaX[0] = deltaX;
      fIds[1] = t2.fId;

      return true;
    } else if (fIds[2] == 0) {      // the second t2 is already set
      //c compatibility
      if (deltaX.Mag2() < sqr(t2.fTime - fMicroSecond))
        return false;
      if ((deltaX - fDeltaX[0]).Mag2() < sqr(t2.fTime - fTimes[0] - fMicroSecond))
        return false;

      fTimes[1] = t2.fTime - fMicroSecond;
      fDeltaX[1] = deltaX;
      fIds[2] = t2.fId;

      const bool lightCone = IsInLightCone(0.);

      if (!lightCone) {
        RemoveT2(3);
        return false;
      }

      return true;
    } else {                        //trying to add more than 3 stations -> false
      return false;
    }
  }

  template<class data>
  bool
  AddT2(const data& t2, int deltaX, int deltaY, int deltaZ)
  {
    if (ContainsId(t2.fId))         //reject same station with later t2 with same id
      return false;

    if (!t2.fId)
      return false;

    if (fIds[1] == 0) {             //only the first T2 is set
      //check compatibility with speed of light
      const int deltaX2 = sqr(deltaX) + sqr(deltaY) + sqr(deltaZ);
      if (deltaX2 < sqr(t2.fTime - fMicroSecond))
        return false;

      fTimes[0] = t2.fTime - fMicroSecond;
      fDeltaX[0].SetXYZ(deltaX, deltaY, deltaZ);
      fIds[1] = t2.fId;

      return true;
    } else if (fIds[2] == 0) {      // the second t2 is already set
      //c compatibility
      const int deltaX2 = sqr(deltaX) + sqr(deltaY) + sqr(deltaZ);
      if (deltaX2 < sqr(t2.fTime - fMicroSecond))
        return false;
      const TVector3 deltaVec(deltaX, deltaY, deltaZ);
      if ((deltaVec - fDeltaX[0]).Mag2() < sqr(t2.fTime - fTimes[0] - fMicroSecond))
        return false;

      fTimes[1] = t2.fTime - fMicroSecond;
      fDeltaX[1] = deltaVec;
      fIds[2] = t2.fId;

      const bool lightCone = IsInLightCone(0.);

      if (!lightCone) {
        RemoveT2(3);
        return false;
      }
      return true;
    } else {                        //trying to add more than 3 stations -> false
      return false;
    }
  }


  bool
  ContainsId(uint id)
    const
  {
    return fIds[0] == id || fIds[1] == id || fIds[2] == id;
  }


  //checks distances of all stations with a 3 micro-s tolerance for compatibility with speed of light
  //updated to use the Set of planes defined by the first two stations
  bool
  IsInLightCone(double tolerance)
  {
    fij = fDeltaX[0]*fDeltaX[1]/sqrt(fDeltaX[0].Mag2()*fDeltaX[1].Mag2());
    fTau1 = double(fTimes[0])/(-fDeltaX[0].Mag());
    fTau2 = double(fTimes[1])/(-fDeltaX[1].Mag());

    fD = 1 - sqr(fij);

    double tmp2Tau1Tau2 = 2*fTau1*fTau2;
    double tmp = sqr(fTau2) + sqr(fTau1);

    return tmp - tmp2Tau1Tau2*fij <= fD + tolerance && fabs(fij) < 0.99;
  }


  void
  GetTestStat(FittedT2& f)
  {
    double tmp2Tau1Tau2 = (2*fTau1*fTau2);
    double tmp = sqr(fTau2) + sqr(fTau1);

    f.fTestStat = tmp - tmp2Tau1Tau2*fij;
    f.fD = fD;
  }


  double
  GetAvgDistance()
    const
  {
    double avgDistance = 0;
    avgDistance += fDeltaX[0].Mag2() + fDeltaX[1].Mag2();
    avgDistance += (fDeltaX[0] - fDeltaX[1]).Mag2();
    avgDistance /= 3.;

    return sqrt(avgDistance);
  }


  //method for reconstructing the triplet analytically
  // if it fails, a start value for numerical minimisation is returned
  // (is not used at the moment due to computation time)
  bool
  GetStartValues(double& uStart, double& vStart)
  {
    for (int i = 0; i < 3; ++i) {
      if (!fIds[i])
        throw std::logic_error("trying to reconstruct incomplete triplet");
    }

    if (fij > 1 || fTau1 > 1. || fTau2 > 1.) {
      fij = fDeltaX[0]*fDeltaX[1]/sqrt(fDeltaX[0].Mag2()*fDeltaX[1].Mag2());
      fTau1 = fTimes[0]/(-fDeltaX[0].Mag());
      fTau2 = fTimes[1]/(-fDeltaX[1].Mag());
      fD = 1 - sqr(fij);
    }

    if (fTau1 > 1 || fTau2 > 1) {
      throw std::logic_error("sublimunal time differences!");
    }

    if (fDeltaX[0].Mag2() < 0.1 ||
        fDeltaX[1].Mag2() < 0.1 ||
       (fDeltaX[0] - fDeltaX[1]).Mag2() < 0.1) {
      throw std::logic_error("invalid distances between stations");
    }

    const double normalisationFactori = 1./fDeltaX[0].Mag();
    const double normalisationFactorj = 1./fDeltaX[1].Mag();
    const TVector3 iVec(fDeltaX[0].x()*normalisationFactori,
                        fDeltaX[0].y()*normalisationFactori,
                        fDeltaX[0].z()*normalisationFactori);
    const TVector3 jVec(fDeltaX[1].x()*normalisationFactorj,
                        fDeltaX[1].y()*normalisationFactorj,
                        fDeltaX[1].z()*normalisationFactorj);

    //prevent bad starting values
    if (fabs(fij) > 0.99) {
      throw std::logic_error("aligned stations!");
    }

    const TVector3 vk = iVec.Cross(jVec);
    const double alpha = (fTau1 - fTau2*fij)/fD;
    const double beta = (fTau2 - fTau1*fij)/fD;
    const double gammaSqr = (1 - (alpha*iVec + beta*jVec).Mag2())/vk.Mag2();

    TVector3 initAxis;
    initAxis += alpha*iVec;
    initAxis += beta*jVec;

    //add sign choice, so that initAxis.z > 0
    if (gammaSqr > 0) {
      if (vk.z() > 0) {
        initAxis += sqrt(gammaSqr)*vk;
      } else {
        initAxis -= sqrt(gammaSqr)*vk;
      }
    } else {
      if (vk.z() > 0) {
        initAxis += sqrt(fabs(gammaSqr))*vk;
      } else {
        initAxis -= sqrt(fabs(gammaSqr))*vk;
      }

      uStart = initAxis.Theta();
      vStart = initAxis.Phi();

      if (std::isnormal(uStart) && std::isnormal(vStart)) {
        return false;
      } else {
        throw std::out_of_range("not normal numbers!");
      }
      return false;
    }

    if (initAxis.z() < 0)
      initAxis *= -1;

    uStart = initAxis.x();
    vStart = initAxis.y();

    return true;
  }


  //reconstruct a triplet
  ReconstructedT2
  ReconstructTriplet(uint GPSSecond)
  {
    double u = 0;
    double v = 0;

    const bool fitSuccess = GetStartValues(u, v);

    if (fitSuccess) {
      ReconstructedT2 r = ReconstructedT2(GPSSecond, fMicroSecond);
      r.fu = u;
      r.fv = v;

      r.fDistance[0] = fDeltaX[0].Mag();
      r.fDistance[1] = fDeltaX[1].Mag();
      r.fDistance[2] = (fDeltaX[0] - fDeltaX[1]).Mag();

      for (int i = 0 ; i < 3; ++i)
        r.fIds[i] = fIds[i];

      return r;
    } else {
      return ReconstructedT2();
    }
  }


  //remove T2 number x and following...
  // removing number 1 does not really make sense
  void
  RemoveT2(int number)
  {
    if (number <= 2) {
      fIds[1] = 0;
      fIds[2] = 0;
    } else if (number == 3) {
      fIds[2] = 0;
    }
  }


  double
  GetTHatZero(double phi,
              double t0,
              const TVector3& deltaX)
    const
  {
    TVector3 axis;
    axis.SetX(1.);
    axis.SetPhi(phi);

    return -axis*deltaX + t0;
  }


  double
  fcnGausThetaPhi(double t0, double phi)
    const
  {
    double f = 0;
    const TVector3 fDeltaX0;

    f += sqr(GetTHatZero(phi, t0, fDeltaX0));
    f += sqr(fTimes[0] - GetTHatZero(phi, t0, fDeltaX[0]));
    f += sqr(fTimes[1] - GetTHatZero(phi, t0, fDeltaX[1]));

    return f;
  }
};
#endif