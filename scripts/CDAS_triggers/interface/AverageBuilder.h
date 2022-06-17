#include <sd/Constants.h>
#include <utl/Accumulator.h>
#include <io/RootOutFile.h>
#include <string>

#ifndef _AverageBuilder_
#define _AverageBuilder_

typedef unsigned int uint;

class AvgValues;

namespace sd{
  struct LowEnergyData;
}

namespace ShortTerm{
  class Analyser;
}
  
struct CorrectionData{
  double fPressureCoefficient[sd::kNStations][5] = { { 0 } };
  double fLinearCoefficientTPMT[sd::kNStations] = { 0 };
  double fConstCoefficientTPMT[sd::kNStations] = { 0 };

  CorrectionData(){}

  CorrectionData(double* pCoeff)
  {
    for (uint i = 0; i < sd::kNStations; ++i) {
      for (uint j =0; j < 5; ++j)
        fPressureCoefficient[i][j] = pCoeff[j];

      fLinearCoefficientTPMT[i] = 0;
      fConstCoefficientTPMT[i] = 0;
    }
  }

  double GetPrediction(uint id, double T)
  {
    return fLinearCoefficientTPMT[id]*T + fConstCoefficientTPMT[id];
  }
};

class AverageBuilder
{
private:
  //keeping the Avg and Variances
  utl::Accumulator::Var fMeanVar[sd::kNStations][5];

  //utl::Accumulator::Var fMeanPMTTemp[sd::kNStations];
  utl::Accumulator::Var fMeanAoP[sd::kNStations];
  utl::Accumulator::Var fMeanPMTBaseline[sd::kNStations][3];
  utl::Accumulator::Var fMeanPeak[sd::kNStations][3];
  bool fJumpFlag[sd::kNStations] = { false };         //only give Analyser new variance estimates if there was no jumping involved
  bool fUnstable[sd::kNStations] = { false };         //same for T2 based exclusion of unstable stations
  bool fUnstableBaseline[sd::kNStations] = { false }; //mark unstable baselines different from T2
  bool fUnstableScaler[sd::kNStations] = { false };   //mark unstable Scaler rates
  utl::Accumulator::Mean fMeanPressure;
  utl::Accumulator::Mean fMeanTemperature;

  //output of AvgValues
  io::RootOutFile<AvgValues> foutAvg;
  std::string fBasename;

  //Parameters for Averaging
  uint fLastGPSSecond = 0;                      //for checks if data input is continuous
  uint fGPSsLastOutput = 0;

  //Analyser needs the variances from fMeanVar estimates.
  ShortTerm::Analyser* fAnalyser = nullptr;
  void TransferVariances() const;

  //Things for corrections (since only for the long scale this is relevant)
  CorrectionData fCorrectionParameter;
  int GetJumpCorrection(double prediction, double rate) const;  //double as AoP-correction (and p-corr.) have to be applied
  //void FillCorrections(AvgValues& outData);

  void FillAverages(AvgValues& outData) const;
  void BuildAvgOut();
  void Clear();

public:
  AverageBuilder(const std::string& basename);
  //~AverageBuilder() {}
  
  void SetAnalyser(ShortTerm::Analyser& a) { fAnalyser = &a; } 
  void SetCorrectionParameter(std::string filename);

  void AddData(const sd::LowEnergyData& data);

  uint fIntervalLength = 122;
  uint fMaxVariance = 100000;
};
#endif