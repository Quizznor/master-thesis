#include <interface/SecondData.h>
#include <sd/LowEnergyData.h>
#include <STAnalysis/StationStatus.h>
#include <STAnalysis/AnalyserStatus.h>
#include <iostream>
#include <stdexcept>

typedef unsigned short ushort;

ClassImp(ShortTerm::AnalyserStatus)

namespace ShortTerm{

  AnalyserStatus::AnalyserStatus(): 
    fStationStatus(sd::kNStations),
    fTriggerDensity(10000, false),
    fParameter{2, 601, 600, 280, 7500, 2, 120*30},
    fHistScalerPValues("LogScalerPValues", "logScalerPValues",250, -15, 0),
    fHistT2PValues("LogT2PValues", "logT2PValues",250, -15, 0),
    fHistGlobalPValues("LogGlobalPValues", "logGlobalPValues",250, -15, 0)
  {
    for (uint i = 0; i < sd::kNStations; ++i)
      fStationStatus[i].fId = i;
    
  }

  AnalyserStatus::~AnalyserStatus()
  {
  }

  StationStatus& 
  AnalyserStatus::operator[] (const uint& index)
  {
    if (index >= sd::kNStations)
      throw std::out_of_range("invalid station Id");
    return fStationStatus[index];
  }

  uint 
  AnalyserStatus::GetNActiveT2() const
  {
    uint nActive = 0;
    for(uint i = 0; i < sd::kNStations; ++i){
      if(fStationStatus[i].fActiveT2)
        ++nActive;
    }

    return nActive;
  }

  TH1F
  AnalyserStatus::GetHistogramCopy(const uint& channel) const
  {
    if (channel == 0)
      return fHistGlobalPValues;
    else if (channel == 1)
      return fHistT2PValues;
    else if (channel == 4)
      return fHistScalerPValues;
    else 
      return TH1F("h","h",100,0,1);
  }

  uint 
  AnalyserStatus::GetNActiveScaler() const
  {
    uint nActive = 0;
    for(uint i = 0; i < sd::kNStations; ++i){
      if(fStationStatus[i].fActiveScaler)
        ++nActive;
    }

    return nActive;
  }

  uint 
  AnalyserStatus::GetNActive() const
  {
    uint nActive = 0;
    for(uint i = 0; i < sd::kNStations; ++i){
      if(fStationStatus[i])
        ++nActive;
    }

    return nActive;
  }

  float
  AnalyserStatus::GetTriggerDensity()
  {
    uint nTrigger = 0;
    for(const auto& x : fTriggerDensity)
      if(x)
        ++nTrigger;

    return nTrigger/10000.;
  }


  double
  AnalyserStatus::GetAvg(const uint& channel, const uint& station) const
  {
    if(channel > 4 || station > sd::kNStations) 
      return -1;

    return fStationStatus[station].GetAvg(channel);
  }

  int 
  AnalyserStatus::SetParameter(const uint& index, const double& value)
  {
    if(index > 6)
      return -1;

    fParameter[index] = value;
    return 0;
  }

  void 
  AnalyserStatus::UpdateStatus(const uint& GPSSecond)
  {
    for(uint i = 0; i < sd::kNStations; ++i)
      fStationStatus[i].UpdateStatus(fParameter, GPSSecond); 
  }

  void 
  AnalyserStatus::UpdateAverages(const sd::LowEnergyData& Data)
  {
    uint nTrueT2 = 0;
    uint nTrueScaler = 0;

    bool T2 = false;
    bool scaler = false;

    for(uint i = 0; i < sd::kNStations; ++i){
      fStationStatus[i].UpdateAverages(Data, T2, scaler);

      if(T2)
        ++nTrueT2;

      if(scaler)
        ++nTrueScaler;
    }

    if(nTrueT2 == 0)
      ++fMissingT2;
    if(nTrueScaler == 0)
      ++fMissingScaler;

    ++fNProcessedTotal;
    this->UpdateStatus(Data.fGPSSecond);
  }

  void
  AnalyserStatus::Reset()
  {
    *this = AnalyserStatus();
  }

  bool 
  AnalyserStatus::IsActive(const uint& channel, const uint& stationIndex) const
  {
    if(stationIndex >= sd::kNStations)
      return false;

    if(channel == 1){
      return fStationStatus[stationIndex].fActiveT2;
    } else if (channel == 4) {
      return fStationStatus[stationIndex].fActiveScaler;
    } else {
      return bool(fStationStatus[stationIndex]);
    }
  }
    
  bool 
  AnalyserStatus::IsActive(const uint& stationIndex) const
  {
    if(stationIndex < sd::kNStations)
      return bool(fStationStatus[stationIndex]);
    else
      return false;
  }

  void
  AnalyserStatus::GetVarianceAges(StatusOutput& status) const
  {
    for (uint i = 0; i < sd::kNStations; ++i) {
      if (status.fGPSSecond - fStationStatus[i].fLastVarianceUpdate > fParameter[6] && fStationStatus[i].fVariance.GetCount()) //only count if data was there
        ++status.fnScalerWithOldVariance;
    }
  }
}