/*
  stores the Event Candidates determined in the first scan on binned data. 
  It is only to be modified by the analyser class, to prevent modification/bias from later correcting/modifying the members.
*/

#ifndef _EvCandidate_
#define _EvCandidate_

#include <Rtypes.h>
#include <sd/Constants.h>
#include <iostream>
#include <vector>

#ifdef __CINT__
#  define ARRAY_INIT_2D
#else
#  define ARRAY_INIT_2D = { { 0 } }
#endif

#ifdef __CINT__
#  define ARRAY_INIT
#else
#  define ARRAY_INIT = { 0 }
#endif

typedef unsigned short ushort;
typedef unsigned int uint;

namespace ShortTerm{
  class Analyser;
}

class EventCandidate
{ 
public:
  uint fEventTime;           // = Trigger Time, i.e. the end of the fnToTSeconds interval if triggered by a ToT1/2
  double fSignificances[5];  //one bin trigger: rvalues to calculate significance (also global), ToT1/2: saves n over threshold for each channel
  double fpValue = 0;        //minimum of all possible p-Values (a.k.a. global if global < scaler || t2 or vice versa)
  int fTriggerPath;          //see scetch (L1-scaler, L1-T2, L2, ToT1, ToT2 -> 1,2,3,4,5; 6 is reserved for the standard constructor a.k.a. empty Candidate)
  ushort fNactiveT2;         //how many stations were used in the analysis for this event?
  ushort fNactiveScaler;
  double fpToT;              //threshold for ToT triggers (default is 1e-3), filled by Analyser with current value
  ushort fnToTSeconds;       //Length of the sliding window for ToTs, filled by Analyser on construction
  double fPull = 0;
  double fAndersonDarling = 0;
  double fpValueAD = -1;
  double fchiSquareOverDoF = 0;
  ushort fnDoF = 0;   

  ushort fnScalersAbove3Sigma = 0;
  std::vector<ushort> fnScalersAbove10Sigma;
  ushort fnT2Above3Sigma = 0;
  std::vector<ushort> fnT2Above10Sigma;

  std::vector<double> fpValuesGlobal;            //for detection of pulses longer than 1 s
  std::vector<double> fpValuesScalers; 
  std::vector<double> fpValuesT2;  

  float fEstimatedSignalT2 = 0;
  float fEstimatedSignalScaler = 0;

  float fData[5][sd::kNStations] ARRAY_INIT_2D ;        //saves data of that second in case of L-triggers (not for ToT's !)
  float fAverages[5][sd::kNStations] ARRAY_INIT_2D ;    //saves averages of that second in case of L-triggers
  float fEstimatedScalerVariance[sd::kNStations] ARRAY_INIT;
  bool fInAnalysis[2][sd::kNStations] ARRAY_INIT_2D;    //save the Status per Station

  EventCandidate();
  EventCandidate(const double& time, const int& triggerpath, const uint& nToTSeconds, const double& ptot);
  
  explicit operator bool() const { return fTriggerPath != 6; }

  void TestUniformity(const uint& channel = 4);
  double AndersonDarlingTest(std::vector<double>& scaledDevFromMean);
  void SetChiSquare();

  void GetAverages(float* OutavgData, const uint& nmax, const uint& channel); //nmax for length of array Out...
  void GetData(float* OutSeconddata, const uint& nmax, const uint& channel); 

  double GetEventTime() const;      // uses front of ToT interval in case of ToT Trigger (as time of ToT is designed when the first sign. bin goes out of the window)
  double GetSignificance(const uint& channel) const;
  double GetRValue(const uint& channel) const;
  int GetTriggerPath() const;
  double GetSignalUncertainty(const uint& channel) const;

  template<typename T>
  void SetSignificances(T* sig)
  {
    for (int i = 0; i < 5; ++i) {
      try{
        fSignificances[i] = sig[i];
      } catch(std::exception& e) {
        fSignificances[i] = -1;
        std::cerr << "error setting significances " << e.what() << std::endl;
      }
    }
  }
  void SavePull(const uint& channel = 4);       //calculates something like a pull-distribution with the stations divided into subarrays (c.f. ROOT-Skripte/OutlierCheck.C and ../msc/msc.pdf (in  Analysis Methods/EventSelection))
  uint GetExcessCount(const uint& channel) const;
  void SaveNaboveXSigma();
  uint CroppedExcess(const uint& channel, const uint& maxVal) const;

  friend class ShortTerm::Analyser;
  friend class Converter; 
  ClassDefNV(EventCandidate, 1);
};

#endif
