#include <utl/Accumulator.h>
#include <vector>
#include <Rtypes.h>

#ifndef _STStationStatus_
#define _STStationStatus_

class SecondData;
namespace sd{
  struct LowEnergyData;
}

typedef unsigned int uint;

namespace ShortTerm{

  struct StationStatus
  {
    StationStatus();
    StationStatus(const uint& Id);
    ~StationStatus();
    
    utl::Accumulator::LowPass fAvgs[5];

    utl::Accumulator::LowPass fVariance;    //save the Variance of the station (scaler data) every time a long term avg is saved.

    std::vector<bool> fT2Monitoring;        //saves if the station had sane data in the last 300 seconds -> if < ... not reliable average => declare inactive
    std::vector<bool> fScalerMonitoring;    

    bool fZeroCount = false;

    bool fActiveT2 = false;
    bool fActiveScaler = false;

    bool fJumpFlag = false;
    bool fUnstable = false;
    bool fUnstableBaseline = false;
    bool fUnstableScaler = false;

    uint fnSecondsProcessedScaler = 0;
    uint fnSecondsProcessedT2 = 0;
    uint fConsecutiveZerosT2 = 0;
    uint fConsecutiveZerosScaler = 0;

    bool fwasActiveT2 = false;
    bool fwasActiveScaler = false;

    uint fLastVarianceUpdate = 0;
    float fvarTempStorage = 0;              //save the variance give to this until Update Status is called and further processed

    uint fId;

    void UpdateAverages(const sd::LowEnergyData&, bool& T2data, bool& Scalerdata);  //returns true if there is data to update the average 
    void UpdateVariance(const float& Var);                                          //give updated variance estimator to stationstatus
    void UpdateStatus(double* Parameter, const uint& GPSSecond);                    //Parameter from Analyser Status (gives information when a station is active/not active)

    void Reset();

    double GetAvg(const uint& channel) const;
    explicit operator bool() const { return fActiveT2 || fActiveScaler; }

    ClassDefNV(StationStatus, 1);
  };

}
#endif