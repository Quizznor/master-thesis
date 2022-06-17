#include <T2Interferometry/TimeStamp.h>
#include <T2Interferometry/ProjectedPositions.h>
#include <iostream>
#include <utl/Accumulator.h>

ClassImp(TimeStamp)

double
TimeStamp::GetAvgDistance(const ProjectedPositions& pos)
const
{
  double avg = 0;
  unsigned n = 0;
  for (uint i = 0; i < fIds.size(); ++i) {
    for (uint j = i; j < fIds.size(); ++j) {
      avg += pos.GetRealDistance(fIds[i], fIds[j]);
      ++n;
    }
  }

  return avg/n;
}

double
TimeStamp::GetChiSquare()
const
{
  double chiSqr = 0;
  for (const auto& tPair : fReconstructedTrefs) {
    chiSqr += pow(tPair.first - fMicroSecond, 2)/pow(tPair.second, 2);
  }

  return chiSqr;
}


//calculate a covariance matrix based on the 
//  sigmaTref^2 = var_uncorr^2 + var_corr^2 assumption (c.f. InterferometricAnalyser.cc)
std::vector<double>
TimeStamp::GetCorrelationMatrixEstimate(const ProjectedPositions& positions)
const
{
  std::vector<double> covarianceMatrix;     //use similar binning as direction bins -> i*(size) + j is entry ij
  covarianceMatrix.resize(fIds.size()*fIds.size(), 0);

  for (uint i = 0; i < fIds.size(); ++i) {
    for (uint j = 0; j < fIds.size(); ++j) {
      const auto delta1 = positions.GetPosition(fIds[i]);
      const auto delta2 = positions.GetPosition(fIds[j]);
      
      covarianceMatrix[i*fIds.size() + j] = delta1*delta2/(delta1.Mag() + delta2.Mag());
    }
  }

  return covarianceMatrix;
}

void
TimeStamp::erase(unsigned first, unsigned last = 0)
{
  if (!last) {
    fReconstructedTrefs.erase(fReconstructedTrefs.begin() + first, fReconstructedTrefs.end());
    fIds.erase(fIds.begin() + first, fIds.end());  
  } else {
    fReconstructedTrefs.erase(fReconstructedTrefs.begin() + first, fReconstructedTrefs.begin() + last);
    fIds.erase(fIds.begin() + first, fIds.begin() + last);
  }  
}

//Need to remove accidental T2's from a time series, that would be compatible
// conerning the 2 sigma requirement
TimeStamp
TimeStamp::RemoveAccidentals(double threshold)
{
  TimeStamp remainingData(*this);
  //Get mean of delta T between times
  utl::Accumulator::Var deltaT;
  std::vector<double> deltaTimes;
  for (uint i = 1; i < fReconstructedTrefs.size(); ++i) {
    double x = fReconstructedTrefs[i].first - fReconstructedTrefs[i - 1].first;
    
    deltaTimes.push_back(x);
    deltaT(x);
  }

  //take median as better estimator for 'normal' distances
  const double median = deltaTimes.size() > 0 ? deltaTimes[deltaTimes.size()/2] : deltaT.GetMean();//deltaT.GetMean();
  const double sigma = sqrt(deltaT.GetVar());

  bool cutted = false;

  for (auto it = deltaTimes.begin(); it != deltaTimes.end(); ++it) {
    //relative cut + hard cut on absolute value
    if ((*it - median)/sigma > threshold || *it > 10) {
      int i = it - deltaTimes.begin() + 1;
      
      erase(i);

      remainingData.erase(0, i);

      cutted = true;
      break;
    }
  }

  fTimeSpread = fReconstructedTrefs.back().first - fReconstructedTrefs.front().first;
  remainingData.fMicroSecond = remainingData.size() > 0 ? remainingData.fReconstructedTrefs[remainingData.size()/2].first : 0;

  if (!cutted)
    remainingData.reset();

  return remainingData;
}