#ifndef _triplet_h_
#define _triplet_h_

#include <t2/T2Data.h>
#include <t2/Vector.h>
#include <utl/Math.h>
#include <iostream>
#include <Rtypes.h>
#include <T2Dump/Utl.h>
#include <T2Dump/Units.h>

namespace t2 {

  struct rTriplet {  //aka reconstructedTriplet
    uint fGPSSecond = 0;
    double ft0 = 0;            //in us!

    T2Data fTrigger[3];

    short fClusterLabel = 0;

    int fAvgDistance = 0;   //! in meter

    uint fIds[3];
    Vector<long> fPositions[3];     //!
    Vector<long> fCenter;           //!
    Vector<double> fAxis;           //!

    Vector<double> fderivativeAxis[3]; //!

    float fu = 2;
    float fv = 2;

    float fsigmaU = 0;              //!
    float fsigmaV = 0;              //!

    explicit operator bool() const { return fGPSSecond; }

    bool operator==(const rTriplet& b) 
      const
    {
      return    fGPSSecond == b.fGPSSecond
             && ft0 == b.ft0
             && fu == b.fu
             && fv == b.fv
             && fTrigger[0] == b.fTrigger[0]
             && fTrigger[1] == b.fTrigger[1]
             && fTrigger[2] == b.fTrigger[2];
    }

    bool operator<(const rTriplet& b) const { return TimeDifference(b) < 0; }
    bool operator>(const rTriplet& b) const { return TimeDifference(b) > 0; }

    bool
    ContainsNN()
      const
    {
      if (crown(fPositions[0].fX, fPositions[1].fX, 
          fPositions[0].fY, fPositions[1].fY) == 1)
        return true;
      if (crown(fPositions[0].fX, fPositions[2].fX, 
          fPositions[0].fY, fPositions[2].fY) == 1)
        return true;
      if (crown(fPositions[2].fX, fPositions[1].fX, 
          fPositions[2].fY, fPositions[1].fY) == 1)
        return true;
      return false;
    }

    uint
    GetNToT()
      const
    {
      uint n = 0;
      for (const auto& t : fTrigger) {
        if (t.IsToT())
          ++n;
      }
      return n;
    }

    double
    GetAvgDistance()
      const
    {
      double avgDistance = sqrt(fPositions[1].Distance2(fPositions[0]));
      avgDistance += sqrt(fPositions[2].Distance2(fPositions[0]));
      avgDistance += sqrt(fPositions[2].Distance2(fPositions[1]));
      return avgDistance;
    }

    void 
    CalculateAvgDistance()
    {
      fAvgDistance = sqrt(fPositions[1].Distance2(fPositions[0]));
      fAvgDistance += sqrt(fPositions[2].Distance2(fPositions[0]));
      fAvgDistance += sqrt(fPositions[2].Distance2(fPositions[1]));
      fAvgDistance /= 3.; 

      for (int i = 0; i < 3; ++i)
        fCenter += fPositions[0];

      fCenter /= 3;
    }

    void 
    SetUncertainties()
    {
      const double det = abs( (fPositions[0].fX - fPositions[1].fX)
                             *(fPositions[0].fY - fPositions[2].fY) 
                             -(fPositions[0].fX - fPositions[2].fX)
                             *(fPositions[0].fY - fPositions[1].fY));

      fsigmaU = kMicroSecond/det*std::sqrt(utl::Sqr(fPositions[0].fY - fPositions[2].fY) 
                                   + utl::Sqr(fPositions[0].fY - fPositions[1].fY));
      fsigmaV = kMicroSecond/det*std::sqrt(utl::Sqr(fPositions[0].fX - fPositions[1].fX) 
                                   + utl::Sqr(fPositions[0].fX - fPositions[2].fX));
    }

    //calculates an estimate of sigma_t with
    // t = t_1 - a*(x_index(t) - x_1)
    // based on the uncorrelated (!) uncertainties of the 3 trigger times
    double 
    GetSigmaTSqr(const rTriplet& t, uint index) 
      const
    {
      double output = 0;
      if (ContainsId(t.fTrigger[index].fId))
        return output;

      Vector<double> deltaX = fPositions[0];
      deltaX -= t.fPositions[index];

      for (uint i = 0; i < 3; ++i) {
        output += utl::Sqr(fderivativeAxis[i]*deltaX + (i == 0)*1);
      }

      return output*kTimeVariance2;
    }

    //tHat: expected time of trigger assuming plane front from *(this) triplet
    double
    GetThat(const rTriplet& b, int index)
      const
    {
      return fAxis*(fPositions[0] - b.fPositions[index]) + fTrigger[0].fTime;
    }

    double
    GetThat(const Vector<long int>& position)
      const
    {
      return fAxis*(fPositions[0] - position) + fTrigger[0].fTime;
    }

    double TimeDifference(const rTriplet& b) const
    { return 1e6*(fGPSSecond - b.fGPSSecond) + ft0 - b.ft0; }

    template<class C>
    double TimeDifference(const C& c) const
    { return 1e6*(fGPSSecond - c.fGPSSecond) + ft0 - c.fMicroSecond; }

    //`old' implementation:
    //  euclidean distance in u,v and scaled in t0 (ignoring uncertainties)
    // new: chi-Square like expression, based on uncertainties from t_i
    //  -> implicit treatment of correlations as the full grad t V (grad t)^t 
    //     expression was used to derive the uncertainties c.f. GAP-2018-XXX
    double 
    Distance2(const rTriplet& b) 
      const
    { 
      if (useOldMetric)
        return   utl::Sqr(fu - b.fu) + utl::Sqr(fv - b.fv) 
               + kTimeDistanceScale2*utl::Sqr(TimeDifference(b)); 

      double chiSquare = 0;
      int nDof = 0;

      for (int i = 0; i < 3; ++i) {
        if (fGPSSecond == b.fGPSSecond &&
            ContainsTrigger(b.fTrigger[i]))
          continue;

        const int deltaGPS = int(b.fGPSSecond) - int(fGPSSecond);
        chiSquare += utl::Sqr(GetThat(b, i) - (b.fTrigger[i].fTime + 
                              kSecond*kMicroSecond*deltaGPS))/
                      (GetSigmaTSqr(b, i) + kTimeVariance2);
        ++nDof;
      }

      //symmetrise
      for (int i = 0; i < 3; ++i) {
        if (fGPSSecond == b.fGPSSecond &&
            b.ContainsTrigger(fTrigger[i]))
          continue;
        const int deltaGPS = int(fGPSSecond) - int(b.fGPSSecond);
        chiSquare += utl::Sqr(b.GetThat(*this, i) - 
                    (fTrigger[i].fTime + kSecond*kMicroSecond*deltaGPS))/
                      (b.GetSigmaTSqr(*this, i) + kTimeVariance2);
        ++nDof;
      }

      return nDof ? chiSquare/nDof : chiSquare;
    }

    bool
    TestReconstruction(const Vector<double>& axis)
      const
    {
      Vector<long> deltaX21 = fPositions[1] - fPositions[0];
      Vector<long> deltaX31 = fPositions[2] - fPositions[0];

      //times are in meter -> 10m tolerance, while 1 us = 300 m
      return (axis*deltaX21 - fTrigger[0].fTime + fTrigger[1].fTime) 
                      < 1 && 
             (axis*deltaX31 - fTrigger[0].fTime + fTrigger[2].fTime) 
                      < 1;
    }

    bool
    TestReconstruction()
      const
    {
      Vector<double> axis(fu, fv, sqrt(1 - fu*fu - fv*fv));
      return TestReconstruction(axis);
    }

    bool 
    ContainsId(uint id) 
      const
    { 
      return fTrigger[0].fId == id || 
             fTrigger[1].fId == id || 
             fTrigger[2].fId == id;
    }

    bool
    ContainsTrigger(const T2Data& t2)
      const
    {
      for (const auto& t : fTrigger) {
        if (t2 == t)
          return true;
      }
      return false;
    }

    rTriplet() = default;
    ~rTriplet() = default;
    rTriplet(const uint gps) : fGPSSecond(gps) { }

    template<class C>
    rTriplet(const C& c) : 
      ft0(c.ft0), fu(c.fu), fv(c.fv) {}


    ClassDefNV(rTriplet, 2);
  };

  inline
  std::ostream&
  operator<<(std::ostream& os, const rTriplet& r)
  {
    return os << r.fu << ' ' << r.fv 
              << ' ' << r.fGPSSecond 
              << ' ' << r.ft0;
  }

  template<typename T>
  struct triplet{
    T2Data fTrigger[3];    //input data
    Vector<T> fPositionFirst;
    Vector<T> fDeltaX12;   //distance vectors in meters (int sufficient)

    triplet() = default;
    ~triplet() = default;
    triplet(T2Data t2) { fTrigger[0] = t2; }

    int 
    crownFromFirst(const Vector<T>& deltaX) 
      const
    { 
      return crown(0., deltaX.fX, 0., deltaX.fY);
    }

    //no abs: if time difference is negative, the sorting is wrong
    bool
    LightCone(int deltaTime, const Vector<T>& deltaX)
      const
    {
      if (deltaTime < abs(deltaX.fX))
        return true;
      if (deltaTime < abs(deltaX.fY))
        return true;
      const int dt2 = utl::Sqr(deltaTime);
      const long xy2 = deltaX.XYMag2();
      if (dt2 < xy2)
        return true;
      if (dt2 > xy2 + utl::Sqr(deltaX.fZ))
        return true;
      return false;
    }

    bool 
    AddSecondT2(const T2Data& t2, const Vector<T>& deltaX)
    {
      if (!fTrigger[0].fId) {
        std::cerr << "AddSecondT2: first station not set!" << std::endl;
        return false;
      }

      if (!t2)
        return false;

      if (ContainsId(t2.fId))
        return false;

      if (!deltaX.Mag2()) //should never happen in real data (or is taken care of with ids)
        return false;

      const int deltaTime = t2.fTime - fTrigger[0].fTime;
      if (LightCone(deltaTime, deltaX)) {
        fDeltaX12 = deltaX;
        fTrigger[1] = t2;
        return true;
      } else {
        return false;
      }
    }

    bool 
    AddThirdT2(rTriplet& reconstructed, 
               const T2Data& t2, 
               const Vector<T> deltaX13)
    {
      if (!fTrigger[0].fId || !fTrigger[1].fId) {
        std::cerr << "Warning(AddThirdT2): first or second"
                     " station not set!" << std::endl;
        return false;
      }

      if (!t2)
        return false;

      if (ContainsId(t2.fId))
        return false;

      const int deltaTime13 = t2.fTime - fTrigger[0].fTime;
      if (!LightCone(deltaTime13, deltaX13))
        return false;
      
      const int deltaTime23 = t2.fTime - fTrigger[1].fTime;
      Vector<T> deltaX23 = fDeltaX12;
      deltaX23 -= deltaX13;

      if (!LightCone(deltaTime23, deltaX23))
        return false;

      if (fDeltaX12.Mag2() < 1. || deltaX13.Mag2() < 1.)
        return false;

      const double invMagI = 1./sqrt(fDeltaX12.Mag2());
      const double invMagJ = 1./sqrt(deltaX13.Mag2());

      const double ij = fDeltaX12*deltaX13*invMagI*invMagJ;
      //avoid degeneracy and numeric instabilities
      if (fabs(ij) > 0.99)
        return false;

      const double tau1 = -(fTrigger[1].fTime - fTrigger[0].fTime)*invMagI;
      const double tau2 = -(t2.fTime - fTrigger[0].fTime)*invMagJ;    
      const double d = 1 - utl::Sqr(ij);

      //plane condition
      if (utl::Sqr(tau1) + utl::Sqr(tau2) > 2*tau1*tau2*ij + d)
        return false;

      const Vector<double> iVec(fDeltaX12.fX*invMagI, 
                                fDeltaX12.fY*invMagI, 
                                fDeltaX12.fZ*invMagI);
      const Vector<double> jVec(deltaX13.fX*invMagJ, 
                                deltaX13.fY*invMagJ, 
                                deltaX13.fZ*invMagJ);
      const Vector<double> vk(iVec.fY*jVec.fZ - iVec.fZ*jVec.fY,
                              iVec.fZ*jVec.fX - iVec.fX*jVec.fZ,
                              iVec.fX*jVec.fY - iVec.fY*jVec.fX);

      const double invD = 1./d;
      const double alpha = (tau1 - tau2*ij)*invD;
      const double beta = (tau2 - tau1*ij)*invD;

      //copies as the originals are needed for sigma_that
      Vector<double> tmpI(iVec);  
      Vector<double> tmpJ(jVec);
      Vector<double> tmpVk(vk);
      tmpI *= alpha;
      tmpJ *= beta;

      tmpI += tmpJ;
      const double gammaSqr = (1 - tmpI.Mag2())/vk.Mag2();

      //sanity check, however plane condition should prevent this in any case
      if (gammaSqr < 0) {
        std::cerr << "reco: this should never happen!"  
                  << gammaSqr << ' '
                  << fTrigger[0].fId << ' '
                  << fTrigger[1].fId << ' '
                  << t2.fId
                  << std::endl;
        return false;
      }
      
      const double gamma = sqrt(gammaSqr);
      tmpVk *= gamma;
      
      //decide on the sign: needs unique choice of sign
      //  in the case, that both solutions are from the same hemisphere,
      //  the triplet is considered incompatible with a plane front moving
      //  with speed of light
      bool negativeGamma = false;
      if (tmpI.fZ + tmpVk.fZ >= 0 && tmpI.fZ - tmpVk.fZ < 0) {
        tmpI += tmpVk;
      } else if ((tmpI.fZ + tmpVk.fZ < 0 && tmpI.fZ - tmpVk.fZ >= 0)) {
        tmpI -= tmpVk;
        negativeGamma = true;
      } else {
        return false;
      }

      //grad_ti of alpha, beta, gamma
      // use t_i as those are independent -> C(t_i, t_j) = delta_ij
      Vector<double> derivativeAlpha(invMagI - ij*invMagJ, -invMagI, ij*invMagJ);
      Vector<double> derivativeBeta(invMagJ - ij*invMagI, ij*invMagI, -invMagJ);
      derivativeAlpha *= invD;
      derivativeBeta *= invD;

      reconstructed.fu = tmpI.fX;
      reconstructed.fv = tmpI.fY;

      if (std::isnan(tmpI.fX) || std::isnan(tmpI.fY)) {
        std::cerr << "reco: Abnomal values (-> u)! "
                  << tmpI.fX << " " << tmpI.fY << " "
                  << invMagI << " " << invMagJ
                  << std::endl;
        return false;
      }

      for (int i = 0; i < 2; ++i) {
        reconstructed.fIds[i] = fTrigger[i].fId;
      }
      reconstructed.fIds[2] = t2.fId;

      reconstructed.fTrigger[0] = fTrigger[0];
      reconstructed.fTrigger[1] = fTrigger[1];
      reconstructed.fTrigger[2] = t2;

      //grad_ti gamma
      Vector<double> derivativeGamma(-2*tau1*invMagI - 2*tau2*invMagJ 
                                       + 2*ij*invMagI*invMagJ*
                                       (2*fTrigger[0].fTime - fTrigger[1].fTime - t2.fTime),
                                     2*tau1*invMagI + 2*ij*invMagI*invMagJ*
                                       (-fTrigger[0].fTime + t2.fTime),
                                     2*ij*invMagI*invMagJ*(-fTrigger[0].fTime + fTrigger[1].fTime)
                                       + 2*tau2*invMagJ);
      derivativeGamma *= negativeGamma ? -1./(2*d*d*gamma) : 1./(2*d*d*gamma);

      const double expectedTimeDifference =   tmpI.fX*fPositionFirst.fX
                                            + tmpI.fY*fPositionFirst.fY
                                            + tmpI.fZ*fPositionFirst.fZ;
      reconstructed.ft0 = (fTrigger[0].fTime + expectedTimeDifference)/kMicroSecond;
      reconstructed.fAxis = tmpI;

      Vector<double> derivativeAxisT1(0, 0, 0);
      derivativeAxisT1 += iVec*derivativeAlpha.fX;
      derivativeAxisT1 += jVec*derivativeBeta.fX;
      derivativeAxisT1 += vk*derivativeGamma.fX;


      Vector<double> derivativeAxisT2(0, 0, 0);
      derivativeAxisT2 += iVec*derivativeAlpha.fY;
      derivativeAxisT2 += jVec*derivativeBeta.fY;
      derivativeAxisT2 += vk*derivativeGamma.fY;

      Vector<double> derivativeAxisT3(0, 0, 0);
      derivativeAxisT3 += iVec*derivativeAlpha.fZ;
      derivativeAxisT3 += jVec*derivativeBeta.fZ;
      derivativeAxisT3 += vk*derivativeGamma.fZ;

      reconstructed.fderivativeAxis[0] = derivativeAxisT1;
      reconstructed.fderivativeAxis[1] = derivativeAxisT2;
      reconstructed.fderivativeAxis[2] = derivativeAxisT3;

      return true;
    }
    
    bool 
    ContainsId(uint id) 
      const
    { 
      return fTrigger[0].fId == id || fTrigger[1].fId == id || fTrigger[2].fId == id;
    }
  };
};
#endif

