#include <interface/AverageBuilder.h>
#include <sd/LowEnergyData.h>
#include <interface/AvgValues.h>
#include <STAnalysis/Analyser.h>
#include <iostream>
#include <fstream>
#include <cmath>

AverageBuilder::AverageBuilder(const std::string& basename) : foutAvg(basename + "_Avg.root"), fBasename(basename)
{
  for (uint i = 0; i < sd::kNStations; ++i)
    fCorrectionParameter.fPressureCoefficient[i][4] = -7.44;
}

void
AverageBuilder::AddData(const sd::LowEnergyData& data)
{
  if (fLastGPSSecond && (int(data.fGPSSecond) - int(fLastGPSSecond) < 0 || data.fGPSSecond - int(fLastGPSSecond) > 1)) {
    std::cerr << "Warning, non continuous data stream!" << std::endl; 
  }

  if (!fGPSsLastOutput) {
    fGPSsLastOutput = data.fGPSSecond - 1;
    fLastGPSSecond = data.fGPSSecond - 1;
  }

  fLastGPSSecond = data.fGPSSecond;

  //Avg-ing
  fMeanPressure(data.fAtmosphere.fPressure);
  fMeanTemperature(data.fAtmosphere.fTemperature);

  for (uint i = 0; i < sd::kNStations; ++i) {
    
    fMeanVar[i][0](data.fStation[i].fTotRate);
    fMeanVar[i][2](data.fStation[i].f70HzRate);
    fMeanVar[i][3](data.fStation[i].fT1Rate);

    if (data.fStation[i].fT2 != 0)
      fMeanVar[i][1](data.fStation[i].fT2);

    if (data.fStation[i].fScaler != 0)
      fMeanVar[i][4](data.fStation[i].fScaler);

    if (data.fStation[i].fAreaOverPeak)
      fMeanAoP[i](data.fStation[i].fAreaOverPeak);

    for (uint j = 0; j < 3; ++j) {
      if (data.fStation[i].fPMTBaseline[j])
        fMeanPMTBaseline[i][j](data.fStation[i].fPMTBaseline[j]);
      if (data.fStation[i].fPeak[j])
        fMeanPeak[i][j](data.fStation[i].fPeak[j]);
    }

    if (data.fStation[i].fJumpFlag)
      fJumpFlag[i] = true;
    if (data.fStation[i].fUnstable)
      fUnstable[i] = true;
    if (data.fStation[i].fUnstableBaseline)
      fUnstableBaseline[i] = true;
    if (data.fStation[i].fUnstableScaler)
      fUnstableScaler[i] = true;
  }
  
  //Check if this should be put out (and be 'sent' to the Analyser)
  if (fLastGPSSecond - fGPSsLastOutput == fIntervalLength) { 
    BuildAvgOut();
    TransferVariances();

    Clear();
    fGPSsLastOutput = fLastGPSSecond;
  }
}

void
AverageBuilder::Clear()
{
  for (uint i = 0; i < sd::kNStations; ++i) {
    fMeanAoP[i].Clear();
    fJumpFlag[i] = false;
    fUnstable[i] = false;
    fUnstableBaseline[i] = false;
    fUnstableScaler[i] = false;

    for (uint j = 0; j < 5; ++j) {
      fMeanVar[i][j].Clear();
      if (j < 3) {
        fMeanPMTBaseline[i][j].Clear();
        fMeanPeak[i][j].Clear();
      }
    }
  }

  fMeanPressure.Clear();
  fMeanTemperature.Clear();
}

void
AverageBuilder::TransferVariances() const
{
  float tmpVar[sd::kNStations];

  for(uint i = 0; i < sd::kNStations; ++i){
    float x = fMeanVar[i][4].GetVar();

    if (std::isnormal(x) 
        && !fJumpFlag[i] 
        && !fUnstable[i]
        && !fUnstableBaseline[i]
        && !fUnstableScaler[i]
        && x < fMaxVariance
        && fMeanVar[i][4].GetCount() > fIntervalLength/2.)
      tmpVar[i] = x;
    else
      tmpVar[i] = 0;  //zeros are not processed in the analyser
  }

  fAnalyser->AddVariances(tmpVar);
}

void
AverageBuilder::BuildAvgOut()
{
  AvgValues avgValues(fGPSsLastOutput + 1, fLastGPSSecond);

  FillAverages(avgValues);
  //this->FillCorrections(avgValues);

  foutAvg << avgValues;
}

void
AverageBuilder::FillAverages(AvgValues& outData) const
{
  outData.fMeanPressure = fMeanPressure.GetMean();
  outData.fMeanTemperature = fMeanTemperature.GetMean();

  for (uint i = 0; i < sd::kNStations; ++i) {
    outData.fJumpFlag[i] = fJumpFlag[i];
    outData.fUnstable[i] = fUnstable[i];
    outData.fUnstableBaseline[i] = fUnstableBaseline[i];
    outData.fUnstableScaler[i] = fUnstableScaler[i];
    outData.fMeanAoP[i] = fMeanAoP[i].GetMean();

    for (uint j = 0; j < 5; ++j) {
      float x = fMeanVar[i][j].GetMean();
      float y = fMeanVar[i][j].GetVar();
      float z = fMeanVar[i][j].GetCount();

      if (std::isnormal(x))
        outData.fRawMean[i][j] = x;
      if (std::isnormal(y))
        outData.fRawVar[i][j] = y;

      if (j < 3) {
        if (std::isnormal(fMeanPMTBaseline[i][j].GetMean()))
          outData.fPMTBaseline[i][j] = fMeanPMTBaseline[i][j].GetMean();

        if (std::isnormal(fMeanPeak[i][j].GetMean()))
          outData.fPeak[i][j] = fMeanPeak[i][j].GetMean();
      }

      if (j == 1 || j == 4)
        outData.fActiveSeconds[i][int(j > 2)] = z;
    }
  }
}

int 
AverageBuilder::GetJumpCorrection(double prediction, double rate) const
{
  if (prediction == 0)
    return 0;

  double difference = rate - prediction;
  if (fabs(difference) > 600)
    return 0;

  bool sign = difference > 0;
  int correction = 0;
  difference = fabs(difference);
  
  while (difference > 97) {
    int lowerBound =  pow(2, floor( log2(fabs(difference)) ) );
    int upperBound = pow(2, ceil(log2(fabs(difference))) );

    if (fabs(difference - lowerBound) > fabs(difference - upperBound)) {
      correction += upperBound;
      difference -= upperBound;
    } else {
      correction += lowerBound;
      difference -= lowerBound;
    }
    if (difference < 0)
      break;

    difference = fabs(difference);
  }

  if (sign)
    return -correction;
  else 
    return correction;
}

/*void
AverageBuilder::FillCorrections(AvgValues& outData)
{
  for (uint i = 0; i < sd::kNStations; ++i) {
    //the fit was made after applying AoP correction and adding pressure term, so use this also here:
    float rate = fMeanVar[i][4].GetMean()*3.5/fMeanAoP[i].GetMean() - fCorrectionParameter.fPressureCoefficient[i][4]*(fMeanPressure.GetMean() - 862.5);
    float prediction = fCorrectionParameter.GetPrediction(i, fMeanPMTTemp[i].GetMean());
    
    outData.fDifferenceToPrediction[i] = rate - prediction;

    if (std::isnormal(rate))
      outData.fJumpCorrection[i] = 0;//this->GetJumpCorrection(prediction, rate);
    outData.fTemperatureCorrection[i] = -prediction;
    outData.fJumpFlag[i] = fJumpFlag[i];
    outData.fUnstable[i] = fUnstable[i];

    float tmp = 3.5/fMeanAoP[i].GetMean();
    if (std::isnormal(tmp))
      outData.fAoPCorrection[i] = tmp;

    for (uint j = 0; j < 5; ++j)
      outData.fPressureCorrection[i][j] = -fCorrectionParameter.fPressureCoefficient[i][j]*(fMeanPressure.GetMean() - 862.5);
  }
}*/

void 
AverageBuilder::SetCorrectionParameter(std::string filename)
{
  std::ifstream in(filename.c_str());

  int id = 0;
  double linCoeff = 0;
  double constCoeff = 0;
  double linCoeffError = 0;
  double constCoeffError = 0;

  while (in >> id >> constCoeff >> linCoeff >> constCoeffError >> linCoeffError) {
    if (linCoeff > 0 
        && linCoeffError/linCoeff < 0.05
        && constCoeff > 1000
        && constCoeff < 3000
        && constCoeffError/constCoeff < 0.05) {

      fCorrectionParameter.fLinearCoefficientTPMT[id] = linCoeff;
      fCorrectionParameter.fConstCoefficientTPMT[id] = constCoeff;
    } else {
      fCorrectionParameter.fLinearCoefficientTPMT[id] = 0;
      fCorrectionParameter.fConstCoefficientTPMT[id] = 0;
    }

  }
}
