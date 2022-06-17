#include <T2Interferometry/ProjectedPositions.h>
#include <fstream>
#include <cmath>
#include <iostream>
#include <datatypes.h>

ProjectedPositions::ProjectedPositions()
  : fHealPixBase(4, RING)
{
  ReadPositions(fFilename, fReferencePoint);
}


ProjectedPositions::ProjectedPositions(std::string file,
   const TVector3& referencePoint) :
  fHealPixBase(4, RING),
  fFilename(file),
  fReferencePoint(referencePoint)  
{
  ReadPositions(fFilename, fReferencePoint);
}

void
ProjectedPositions::ReadPositions(std::string file,
   const TVector3& referencePoint)
{
  fFilename = file;
  fReferencePoint = referencePoint;

  ifstream inPositions(file);
  double x = 0;
  double y = 0;
  double z = 0;
  int id = 0;

  double tmp = 0;
  //read in converted to micro seconds
  while (inPositions >> id >> y >> x >> z 
          >> tmp >> tmp >> tmp 
          >> tmp >> tmp >> tmp) {
    TVector3 position(x/300., y/300., z/300.);
    TVector3 deltaX = position - referencePoint;
    fRealPositions[id] = deltaX;

    for (int i = 0; i < fHealPixBase.Npix()/2 + fHealPixBase.Nside()*2; ++i) {
      TVector3 axis(1., 0., 0.);
      const auto direction = fHealPixBase.pix2ang(i);
      
      axis.SetTheta(direction.theta);
      axis.SetPhi(direction.phi);
      fProjectedTimeDifferences[id].push_back(axis*deltaX);
      fSigmaTrefEstimates[id].push_back(GetSigmaTref(fSigmaTrefEstimates[id].size(), id));
    }
  }
}

void
ProjectedPositions::SetOrder(int order)
{
  if (order == fHealPixBase.Order())
    return;

  fHealPixBase.Set(order, RING);

  for (int i = 0; i < 2000; ++i) {
    fProjectedTimeDifferences[i].clear();
    fSigmaTrefEstimates[i].clear();
  }

  ReadPositions(fFilename, fReferencePoint);
}

/*std::vector<unsigned>
ProjectedPositions::GetNeighbouringBins(unsigned bin)
{
  std::vector<unsigned> bins;
  if (bin >= fnBinsPhi*fnBinsCosTheta)
    return bins;

  uint binCosTheta = bin/fnBinsPhi;
  uint binPhi = bin % fnBinsPhi;

  //neighbouring in phi
  if (!binPhi) {
    bins.push_back(bin + 1);
    bins.push_back(bin + fnBinsPhi - 1);
  } else if (binPhi == fnBinsPhi - 1) {
    bins.push_back(bin - 1);
    bins.push_back(bin + 1 - fnBinsPhi);
  } else {
    bins.push_back(bin - 1);
    bins.push_back(bin + 1);
  }

  //neighbouring in cos Theta
  if (!binCosTheta) {
    bins.push_back(bin + fnBinsPhi);
  } else if (binCosTheta == fnBinsCosTheta - 1) {
    bins.push_back(bin - fnBinsPhi);
  } else {
    bins.push_back(bin + fnBinsPhi);
    bins.push_back(bin - fnBinsPhi);
  }

  return bins;
}*/

//to save computation time, the bin size is esimated as average bin size over the
// sphere. For more precision the max_pixrad method can be used.
// It computes the maximal binsize in rad for a given ring. The ring is computed
// from the bin number
double
ProjectedPositions::GetBinWidth(int bin)
const
{
  //return fHealPixBase.max_pixrad(fHealPixBase.pix2ring(bin));
  return sqrt(4*3.1415/fHealPixBase.Npix());
  /*if (axis == 1) {
    return 1./fnBinsCosTheta;
  } else if (axis == 2) {
    return 2*3.1415/fnBinsPhi;
  } else {
    return -1;
  }*/
}

//Get the variance estimate for the quantity \vec{axis}*\vec{delta X} 
// based on Gaussian error propagation and assumption of uniform distributions inside bins
// as bin width in both direction GetBinWidth is used. For details see above
double 
ProjectedPositions::GetVarAX(unsigned bin, const TVector3& deltaX)
const
{
  const auto binCenter = GetBinCenter(bin);
  
  const double cosPhi = cos(binCenter.second);
  const double sinPhi = sin(binCenter.second);

  const double cosTheta = cos(binCenter.first);
  const double sinTheta = sin(binCenter.first);

  double tmp = 0;

  tmp += pow(cosTheta*(cosPhi*deltaX.x() + sinPhi*deltaX.y())*GetBinWidth(bin),2)/12.;
  tmp += pow(sinTheta*(cosPhi*deltaX.y() - sinPhi*deltaX.x())*GetBinWidth(bin),2)/12.;

  /*tmp += pow(deltaX.z() - cosTheta/sinTheta
              *(cosPhi*deltaX.x() + sinPhi*deltaX.y()), 2);
  tmp *= pow(GetBinWidth(bin), 2)/12.;           //var of uniform distr. = BinWidth^2/12

  tmp += sinTheta*sinTheta*
          pow((cosPhi*deltaX.y() - sinPhi*deltaX.x())*GetBinWidth(bin), 2)/12.;*/

  return tmp;
}

double
ProjectedPositions::GetVarAXEstimate(unsigned bin, unsigned id)
const
{
  if (fSigmaTrefEstimates[id].size() == fnBinsCosTheta*fnBinsPhi)
    return fSigmaTrefEstimates[id][bin];

  const TVector3 deltaX = fRealPositions[id] - fReferencePoint;

  return GetVarAX(bin, deltaX);
}

//returns the (gaussian) estimate of the sigma of t_ref = t - \vec{a}*vec{deltaX}
//sigma of t is 1/sqrt(12) from uniform distribution
double
ProjectedPositions::GetSigmaTref(unsigned bin, unsigned id)
const
{
  return sqrt(GetVarAXEstimate(bin, id) + 1/12.); 
}

double
ProjectedPositions::GetRealDistance(uint id1, uint id2)
const
{
  return (fRealPositions[id1] - fRealPositions[id2]).Mag();
}

/*void 
ProjectedPositions::SetNBins(unsigned nCosTheta, unsigned nPhi)
{
  if (nCosTheta == fnBinsCosTheta && nPhi == fnBinsPhi)
    return;

  fnBinsCosTheta = nCosTheta;
  fnBinsPhi = nPhi;

  for (int i = 0; i < 2000; ++i) {
    fProjectedTimeDifferences[i].clear();
    fSigmaTrefEstimates[i].clear();
  }

  ReadPositions(fFilename, fReferencePoint);
}*/

/*int
ProjectedPositions::GetBin(double cosTheta, double phi)
const
{
  if (cosTheta > 1 || cosTheta < 0)
    return -1;
  if (phi > 2*3.1415 || phi < 0)
    return -1;

  return int(cosTheta*fnBinsCosTheta)*fnBinsPhi + int(phi/(2*3.1415)*fnBinsPhi);
}*/

std::pair<double, double>
ProjectedPositions::GetBinCenter(unsigned binNumber)
const
{
  //return std::make_pair((binNumber/fnBinsPhi + 0.5)*1./fnBinsCosTheta,
  //             (binNumber % fnBinsPhi + 0.5)*1./fnBinsPhi*2*3.1415);
  const auto direction = fHealPixBase.pix2ang(binNumber);
  return std::make_pair(direction.theta, direction.phi);
}