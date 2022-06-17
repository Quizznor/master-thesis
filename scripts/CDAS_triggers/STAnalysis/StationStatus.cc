#include <STAnalysis/StationStatus.h>
#include <interface/SecondData.h>
#include <sd/LowEnergyData.h>
#include <utl/Accumulator.h>

ClassImp(ShortTerm::StationStatus)

namespace ShortTerm{

  StationStatus::StationStatus(const uint& Id) : 
    fVariance(2),
    fT2Monitoring(300, false),
    fScalerMonitoring(300, false),
    fId(Id)
  {
  }

  StationStatus::StationStatus() : 
    fVariance(2),
    fT2Monitoring(300, false),
    fScalerMonitoring(300, false),
    fId(0)
  {
  }

  StationStatus::~StationStatus()
  {
  }


  /*
    Updates the average estimates.
     Excludes data if it is more than 10 sigma away, if the station is not marked as unstable (by any of the flags)
    If a station is flagged as unstable the value will be use in anyway to ensure that the mean will catch up with a new value.
    The timescale of the LowPass is about 30s, so excluding at least 90 s from the analysis by flagging it, will result in a good description of 
    the new value by the mean.
    This exclusion is to be placed in a step before this class (c.f. DataCleaner.h/cc)
  */
  void
  StationStatus::UpdateAverages(const sd::LowEnergyData& data, bool& T2data, bool& scalerdata)
  {
    ushort Scaler = data.fStation[fId].fScaler;
    ushort T2 = data.fStation[fId].fT2;
    float ToT = data.fStation[fId].fTotRate;
    float calT = data.fStation[fId].f70HzRate;
    float T1 = data.fStation[fId].fT1Rate;

    fJumpFlag = data.fStation[fId].fJumpFlag;
    fUnstable = data.fStation[fId].fUnstable;
    fUnstableBaseline = data.fStation[fId].fUnstableBaseline;
    fUnstableScaler = data.fStation[fId].fUnstableScaler;

    if (ToT)
      fAvgs[0](ToT);

    if (T2) {
      if (fAvgs[1].GetCount() < 100)
        fAvgs[1](T2);
      else if (fabs(T2 - fAvgs[1].GetMean())/sqrt(fAvgs[1].GetMean()) < 10)
        fAvgs[1](T2);
    }

    if (calT)
      fAvgs[2](calT);
    if (T1)
      fAvgs[3](T1);

    if (Scaler) {
      if (fUnstable || 
          fJumpFlag || 
          fUnstableBaseline || 
          fUnstableScaler ||
          !fActiveScaler) {
        fAvgs[4](Scaler);
      } else if (fabs(Scaler - fAvgs[4].GetMean())/sqrt(fVariance.GetMean()) < 10) {
        fAvgs[4](Scaler);
      }
    }

    //Fill the Monitoring of the data stream, count if there are consecutive zeros.
    if(T2 == 0){
      ++fConsecutiveZerosT2;
      fT2Monitoring[data.fGPSSecond % 300] = false;
      T2data = false;
    } else {
      fConsecutiveZerosT2 = 0;
      fT2Monitoring[data.fGPSSecond % 300] = true;
      T2data = true;
    }
    if(Scaler == 0){
      ++fConsecutiveZerosScaler;
      fScalerMonitoring[data.fGPSSecond % 300] = false;
      scalerdata = false;
    } else {
      fConsecutiveZerosScaler = 0;
      fScalerMonitoring[data.fGPSSecond % 300] = true;
      scalerdata = true;
    }
  }

  /*
    This method decides, when a station can be used in the analysis.
     The parameters are defined in AnalyserStatus and thus can be changed for all stations simultaniously.

     The variance estimate of the scalers is also checked

     meaning of Parameter[] (c.f. AnalyserStatus.h, defined once for all stations, can be modified via AnalyserStatus::SetParameter()):
                                 // 0: NZero till Inactive
                                 // 1: N till Initialised (= active)
                                 // 2: N till 'never was active',
                                 // 3: minimal number of seconds with data wihtin the last 300s to be active/reliable averaged
                                 // 4: maximal variance allowed for useful reconstruction
                                 // 5: maximal ratio for a variance value to the mean (var-estimate/variance.Mean < ...)
                                 // 6: maximal age of variance estimates to be considered active 
  */ 
  void
  StationStatus::UpdateStatus(double* parameter, const uint& GPSSecond)
  {

    //Scaler Part
    if (fConsecutiveZerosScaler > parameter[0]) {        //after .. seconds without data declare as inactive
      fActiveScaler = false;
    } else if (fnSecondsProcessedScaler > parameter[1]) {//after .. seconds with data declare as initialised => active
      fActiveScaler = true;
      fwasActiveScaler = true;
      ++fnSecondsProcessedScaler;
    } else if (!fActiveScaler && fwasActiveScaler && fConsecutiveZerosScaler == 0) {  //returning station => directly initialised
      fActiveScaler = true;
      fwasActiveScaler = true;
      ++fnSecondsProcessedScaler;
    } else if (fConsecutiveZerosScaler > parameter[2]) {    //out of service after .. seconds (old average possibly wrong)
      fwasActiveScaler = false;
      fnSecondsProcessedScaler = 0; 
    } else if (fConsecutiveZerosScaler == 0) {
      fActiveScaler = false;
      ++fnSecondsProcessedScaler;
    }

    //Process possible new variance-estimate
    if (fvarTempStorage) {
      if (fVariance.GetCount() > 5) {
        if (fvarTempStorage < parameter[5]*fVariance.GetMean() && fvarTempStorage < parameter[6]) {
          fVariance(fvarTempStorage);
          fLastVarianceUpdate = GPSSecond;
          fvarTempStorage = 0;
        }  
      } else if (fvarTempStorage < parameter[4]) {
        fVariance(fvarTempStorage);
        fLastVarianceUpdate = GPSSecond;
        fvarTempStorage = 0;
      }
    }

    //Check if the variance estimate of the station is useful:
    // - at least 20 values (corresponds to 120*20s of data)
    // - the estimate is not 0, nan, or inf
    // - the variance is in a reasonable range (standard: < 7500 Hz^2)
    // - the variance estimate isn't too old (default is 30*120s = 30 intervals of avg = 1 h)
    if (fVariance.GetCount() < 20) {
      fActiveScaler = false;
    } else if (!std::isnormal(fVariance.GetMean())) {
      fActiveScaler = false;
    } else if (fVariance.GetMean() > parameter[4]) {
      fActiveScaler = false;
    } else if (GPSSecond - fLastVarianceUpdate > parameter[6]) {
      fActiveScaler = false;
    }


    //T2 Part
    if(fConsecutiveZerosT2 > parameter[0]){   //after .. seconds without data declare as inactive
      fActiveT2 = false;
    } else if (fnSecondsProcessedT2 > parameter[1]){//after .. seconds with data declare as initialised => active
      fActiveT2 = true;
      fwasActiveT2 = true;
      ++fnSecondsProcessedT2;
    } else if (!fActiveT2 && fwasActiveT2 && fConsecutiveZerosT2 == 0) {  //returning station => directly initialised
      fActiveT2 = true;
      fwasActiveT2 = true;
      ++fnSecondsProcessedT2;
    } else if (fConsecutiveZerosT2 > parameter[2]) {    //out of service after .. seconds (old average possibly wrong)
      fwasActiveT2 = false;
      fnSecondsProcessedT2 = 0; 
    } else if (fConsecutiveZerosT2 == 0){
      fActiveT2 = false;
      ++fnSecondsProcessedT2;
    }

    //Check the Monitoring = steadyness of data
    uint nDataT2 = 0;
    uint nDataScaler = 0;

    for(uint i = 0; i < fT2Monitoring.size(); ++i){
      if(fT2Monitoring[i])
        ++nDataT2;
      if(fScalerMonitoring[i])
        ++nDataScaler;
    }

    //reset and reinitialise because avg may be wrong
    if(nDataT2 < parameter[3]){
      fwasActiveT2 = false;
      fActiveT2 = false;
    }
    if(nDataScaler < parameter[3]){
      fwasActiveScaler = false;
      fActiveScaler = false;
    }
  }

  void
  StationStatus::UpdateVariance(const float& var)
  {
    fvarTempStorage = var;
  }

  void
  StationStatus::Reset()
  {
    *this = StationStatus(fId);
  }

  double
  StationStatus::GetAvg(const uint& channel) const
  {
    if(channel > 4)
      return -1;

    if(channel == 2 || channel == 3)
      return fAvgs[channel].GetMean()/10.;

    return fAvgs[channel].GetMean();
  }
}