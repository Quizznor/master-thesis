#include <interface/EventCandidate.h>
//#include <Analyser/AnalyserStatus.h>
//#include <interface/SecondData.h>
#include <utl/Accumulator.h>
#include <random>
#include <chrono>
#include <iostream>
#include <TMath.h>
#include <exception>
#include <Math/ProbFunc.h>

#define nAverage 250
#define nBins	64

#ifdef __CINT__
#  define ARRAY_INIT
#else
#  define ARRAY_INIT = { 0 }
#endif

ClassImp(EventCandidate)

template<typename T>
T
sqr(T a)
{
  return a*a;
}

EventCandidate::EventCandidate():
	fEventTime(0),
	fSignificances{1, 1, 1, 1, 1},
	fTriggerPath(6)
{
}

EventCandidate::EventCandidate(const double& time, const int& trigger, const uint& nToT, const double& ptot):
	fEventTime(time),
  fTriggerPath(trigger),
  fpToT(ptot),
  fnToTSeconds(nToT)
{
}

double
EventCandidate::GetEventTime() const
{
	if (fTriggerPath < 4) {
		return fEventTime;
	} else {
		return fEventTime - fnToTSeconds;
	}
}


double
EventCandidate::GetSignificance(const uint& channel) const
{
	if (channel > 5)
		return -1;

	if (fTriggerPath < 4) {
		return erfc(sqrt(fSignificances[channel]/2.));
	} else {
		return TMath::Binomial(fnToTSeconds, fSignificances[channel])*pow(fpToT, fSignificances[channel]);
	}
}

double
EventCandidate::GetRValue(const uint& channel) const
{
	if (channel > 4)
		return -1;

	return fSignificances[channel];
}

int
EventCandidate::GetTriggerPath() const
{
	return fTriggerPath;
}

/*
  channel: 0 - global likelihood (T2 + Scaler)
  channel: 1 - T2
  channel: 4 - Scaler

  Based on:
    - 0:

    - 1: Based on T2-poissonian, c.f. msc.pdf (series expansion of lambert, sqrt(N) - 1/3.)

    - 4: Calculation of Delta log L = 1/2. with the Likelihood given in Analyser.cc
*/
double
EventCandidate::GetSignalUncertainty(const uint& channel) const
{
  if (channel == 4) {
    double delta = 0;           //sum of delta_i := (N_i - l_i)/sigma_i^2
    double invSigmaSquare = 0;  //sum of 1/sigma^2

    for (uint i = 0; i < sd::kNStations; ++i) {
      if (fInAnalysis[1][i]) {
        delta += (fData[4][i] - fAverages[4][i])/fEstimatedScalerVariance[i];
        invSigmaSquare += 1/fEstimatedScalerVariance[i];
      }
    }

    double b = - delta + fEstimatedSignalScaler*invSigmaSquare;
    double a = invSigmaSquare/2.;

    double delta1 = (-b - sqrt(b*b + 2*a))/(2*a);
    double delta2 = (-b + sqrt(b*b + 2*a))/(2*a);

    if (delta1 > 0)
      return delta1;
    else
      return delta2;

  } else if (channel == 1) {
    int N = 0;
    int active = 0;
    for (uint i = 0; i < sd::kNStations; ++i) {
      if (fInAnalysis[1]) {
        N += (fData[1][i]);
        ++active;
      }
    }

    return (sqrt(N) - 1/3.)/active;
  } else if (channel == 0) {
    return 0;
  } else {
    return 0;
  }

}

void
EventCandidate::GetData(float* OutSeconddata, const uint& nmax, const uint& channel)
{
	if (fTriggerPath > 3) {
		std::cerr << "Warning: Data not set!" << std::endl;
		return;
	}

	for (uint i = 0; i < nmax; ++i) {
		OutSeconddata[i] = fData[channel][i];
	}
}

void
EventCandidate::GetAverages(float* OutAvgs, const uint& nmax, const uint& channel)
{
	if (fTriggerPath > 3) {
		std::cerr << "Warning: Data not set!" << std::endl;
		return;
	}

	for (uint i = 0; i < nmax; ++i) {
		OutAvgs[i] = fAverages[channel][i];
	}
}

void
EventCandidate::SaveNaboveXSigma()
{
  for (uint i = 0; i < sd::kNStations; ++i) {
    float devScalerSigma = 0;
    float devT2Sigma = 0;

    if (fData[4][i] && fInAnalysis[1][i])
      devScalerSigma = fabs(fData[4][i] - fAverages[4][i])/sqrt(fEstimatedScalerVariance[i]);

    if (fData[1][i] && fInAnalysis[0][i])
      devT2Sigma = fabs(fData[1][i] - fAverages[1][i])/sqrt(fAverages[1][i]);

    if (devScalerSigma > 10) {
      ++fnScalersAbove3Sigma;
      fnScalersAbove10Sigma.push_back(i);
    } else if (devScalerSigma > 3) {
      ++fnScalersAbove3Sigma;
    }

    if (devT2Sigma > 10) {
      ++fnT2Above3Sigma;
      fnT2Above10Sigma.push_back(i);
    } else if (devT2Sigma > 3) {
      ++fnT2Above3Sigma;
    }
  }
}



void
EventCandidate::SetChiSquare()
{
  if (fTriggerPath == 1) {
    double chiSquare = 0;
    int nDof = 0;
    for (uint i = 0; i < sd::kNStations; ++i) {
      if (fInAnalysis[1][i]) {
        chiSquare += sqr(fData[4][i] - (fAverages[4][i] + fEstimatedSignalScaler))/(fEstimatedScalerVariance[i] + fEstimatedSignalScaler);
        ++nDof;
      }
    }
    fchiSquareOverDoF = chiSquare/nDof;
    fnDoF = nDof;
  } else if (fTriggerPath == 2) {
    double chiSquare = 0;
    int nDof = 0;
    for (uint i = 0; i < sd::kNStations; ++i) {
      if (fInAnalysis[0][i]) {
        chiSquare += sqr(fData[1][i] - (fAverages[1][i] + fEstimatedSignalT2))/(fAverages[1][i] + fEstimatedSignalT2);
        ++nDof;
      }
    }
    fchiSquareOverDoF = chiSquare/nDof;
    fnDoF = nDof;
  } else if (fTriggerPath == 3) {
    double chiSquare = 0;
    int nDof = 0;
    for (uint i = 0; i < sd::kNStations; ++i) {
      if (fInAnalysis[0][i]) {
        chiSquare += sqr(fData[1][i] - (fAverages[1][i] + fEstimatedSignalT2))/(fAverages[1][i] + fEstimatedSignalT2);
        ++nDof;
      }
      if (fInAnalysis[1][i]) {
        chiSquare += sqr(fData[4][i] - (fAverages[4][i] + fEstimatedSignalScaler))/(fEstimatedScalerVariance[i] + fEstimatedSignalScaler);
        ++nDof;
      }
    }
    fchiSquareOverDoF = chiSquare/nDof;
    fnDoF = nDof;
  }
}

void
EventCandidate::SavePull(const uint& channel)
{
	unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937_64 random(seed1);

  std::uniform_int_distribution<> unif(0, nBins-1);

  utl::Accumulator::Mean meanPull;

  double totalAvg = 0;
  double totalCounts = 0;
  uint nStations = 0;

  uint rebinnedData[nBins] ARRAY_INIT;
  float avgRebin[nBins] ARRAY_INIT;

  for (uint j = 0; j < nAverage; ++j) {
    uint stationInBin[nBins] = { 0 };

    for (uint i = 0; i < sd::kNStations; ++i) {
      if (fData[channel][i] == 0 || !fInAnalysis[int(channel > 1)][i])
        continue;

      uint index = unif(random);
      if (j == 0) {
     		++nStations;
        totalCounts += fData[channel][i];
        totalAvg += fAverages[channel][i];
      }

      avgRebin[index] += fAverages[channel][i];
      rebinnedData[index] += fData[channel][i];
      ++stationInBin[index];
    }

    double s = (totalCounts - totalAvg)/nStations;

    //calculating signal with uncert. in each rebinned region
    double signalValues[nBins] ARRAY_INIT;
    double signalErr[nBins] ARRAY_INIT;

    for (uint i = 0; i < nBins; ++i) {
      signalValues[i] = (rebinnedData[i] - avgRebin[i]);
      signalErr[i] = sqrt(rebinnedData[i]) - 1/3.;    //series expansion of LambertW, see notes (17.1.17)
    }

    double pull = 0;
    for (uint i = 0; i < nBins; ++i) {
      pull += (signalValues[i] - s*stationInBin[i])/(signalErr[i]);
    }

    //reseting arrays
    for (uint i = 0; i < nBins; ++i) {
      avgRebin[i] = 0;
      rebinnedData[i] = 0;
      stationInBin[i] = 0;
    }

    meanPull(pull);
  }

  fPull = meanPull.GetMean();
}

void
EventCandidate::TestUniformity(const uint& channel)
{
  std::vector<double> scaledDevFromMean;

  for (int i = 0; i < sd::kNStations; ++i) {
    if (channel == 4) {
      if (fInAnalysis[1][i]) {
        scaledDevFromMean.push_back((fData[4][i] - fAverages[4][i])/sqrt(fEstimatedScalerVariance[i]));
      }
    } else if (channel == 1) {
      if (fInAnalysis[0][i])
        scaledDevFromMean.push_back((fData[1][i] - fAverages[1][i])/sqrt(fAverages[1][i]));
    }
  }

  fAndersonDarling = this->AndersonDarlingTest(scaledDevFromMean);
  if (fAndersonDarling < 0.4)
    fpValueAD = 0.6;
  else {
    fpValueAD = 1 - sd::kCDFconst - 100*sd::kexpConst/sd::kexpSlope*(exp(sd::kexpSlope*fAndersonDarling) - exp(sd::kexpSlope*0.4));
    if (fpValueAD < 0)
      fpValueAD = 0;
  }
}

double
EventCandidate::AndersonDarlingTest(std::vector<double>& scaledDevFromMean)
{
  std::vector<double> normalizedValues;
  sort(scaledDevFromMean.begin(), scaledDevFromMean.end());

  //get mean, var
  double mean = 0;
  double var = 0;
  uint n = scaledDevFromMean.size();

  for (uint i = 0; i < scaledDevFromMean.size(); ++i) {
    mean += scaledDevFromMean[i];
    var += scaledDevFromMean[i]*scaledDevFromMean[i];
  }

  var = (var - mean/n)/(n - 1);
  mean /= scaledDevFromMean.size();

  for (uint i = 0; i < scaledDevFromMean.size(); ++i) {
    normalizedValues.push_back((scaledDevFromMean[i] - mean)/sqrt(var));
  }

  double Asquared = 0;

  for (uint i = 0; i < n; ++i) {
    Asquared += (2*(i + 1) - 1)*log(ROOT::Math::normal_cdf(normalizedValues[i])) + (2*(n - i - 1) + 1)*log(ROOT::Math::normal_cdf_c(normalizedValues[i]));
  }

  Asquared /= n;

  Asquared += n;

  return -Asquared;
}


uint
EventCandidate::GetExcessCount(const uint& channel) const
{
  uint excess = 0;

  for (uint i = 0; i < sd::kNStations; ++i) {
    if (fData[channel][i] != 0) {
      excess += fData[channel][i] - fAverages[channel][i];
    }
  }

  return excess;
}

uint
EventCandidate::CroppedExcess(const uint& channel, const uint& maxValue) const
{
  uint excess = 0;
  uint nRejected = 0;

  for (uint i = 0; i < sd::kNStations; ++i) {
    if (fData[channel][i] != 0) {
      if (fData[channel][i] >= maxValue) {
        ++nRejected;
      } else {
        excess += fData[channel][i] - fAverages[channel][i];
      }
    }
  }
  std::cout << "nRejected: " << nRejected << std::endl;
  return excess;
}

#undef ARRAY_INIT
#undef nBins
#undef nAverage
