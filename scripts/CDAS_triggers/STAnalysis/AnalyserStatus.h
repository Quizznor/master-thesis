/*
  Saves the current status of the analysis, including Averages, and processed Seconds per stations, what stations are active , ...
*/  
#ifndef _STAnalyserStat_
#define _STAnalyserStat_

#include <vector>
#include <sd/Constants.h>
#include <STAnalysis/StationStatus.h>
#include <interface/StatusOutPut.h>
#include <Rtypes.h>
#include <TH1F.h>

#ifdef __CINT__
#  define ARRAY_INIT
#else
#  define ARRAY_INIT = { 0 }
#endif

typedef unsigned int uint;

namespace sd{
  struct LowEnergyData;
}

namespace ShortTerm{
  class AnalyserStatus
  {
  private:
    std::vector<StationStatus> fStationStatus;
    std::vector<bool> fTriggerDensity;

    uint fNProcessedTotal = 0;

    uint fMissingT2 = 0;
    uint fMissingScaler = 0;

    double fParameter[7];        // 0: NZero till Inactive
                                 // 1: N till Initialised (= active)
                                 // 2: N till 'never was active',
                                 // 3: minimal number of seconds with data wihtin the last 300s to be active/reliable averaged
                                 // 4: maximal variance allowed for useful reconstruction
                                 // 5: maximal ratio for a variance value to the mean (var-estimate/variance.Mean < ...)
                                 // 6: maximal age of variance estimates to be considered active
    void UpdateStatus(const uint& GPSSecond);
    TH1F fHistScalerPValues;
    TH1F fHistT2PValues;
    TH1F fHistGlobalPValues;

  public:
    AnalyserStatus();
    ~AnalyserStatus();
      
    void UpdateAverages(const sd::LowEnergyData&);    //self explanatory, method: 'low pass filter'
    double GetAvg(const uint& channel, const uint& stationIndex) const;

    StationStatus& operator[](const uint& index);
    int SetParameter(const uint& index, const double& value);

    bool IsActive(const uint& channel, const uint& stationIndex) const;
    bool IsActive(const uint& stationIndex) const;
    uint GetNActive() const;
    uint GetNActiveT2() const;
    uint GetNActiveScaler() const;
    void Reset();            //in case everything needs to be restarted after e.g. some missing data
    float GetTriggerDensity();
    void GetVarianceAges(StatusOutput& status) const;

    TH1F GetHistogramCopy(const uint& channel) const;

    friend class Analyser;
    friend class DataCleaner;
    friend class DataCuts;
    ClassDefNV(AnalyserStatus, 1);
  };
}

#endif