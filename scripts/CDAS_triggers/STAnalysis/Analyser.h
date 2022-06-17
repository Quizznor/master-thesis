#ifndef _STAnalyser_
#define _STAnalyser_

#include <STAnalysis/AnalyserStatus.h>
#include <interface/StatusOutPut.h>
#include <io/RootOutFile.h>
#include <sd/Constants.h>
#include <interface/EventCandidate.h>
#include <vector>
#include <string>
#include <utility>

/*
  Main class for 1s based analysis of data.
  Forms "Triggers" and saves EventCandidates
*/
typedef unsigned char uchar;
typedef unsigned int uint;

namespace sd{
  struct LowEnergyData;
}

namespace ShortTerm{

  class Analyser
  {
  private:
    //io parameters/streams
    io::RootOutFile<EventCandidate> foutEvents;
    std::string foutBaseName;

    AnalyserStatus fStatus;

    uint fLastGPSsecond = 0;
    
    //Parameters for Triggering
    double fmaxTriggerP;                           //value for Threshold (L) triggers
    double fTriggerPToT;                           //value for ToT Triggers (ToT i)
    const uint fnToTSeconds = 10;                  //size of the sliding window for ToT generation
    double fMaxRatioToMeanVariance = 2;            //at most ...*fVariance.GetMean() of current variance estimate (usual values: 3000 -> max ~4500)

    //pValues of last fnToTSeconds to create ToT triggers
    std::vector<double> fpValuesGlobal;            //for detection of pulses longer than 1 s
    std::vector<double> fpValuesScalers; 
    std::vector<double> fpValuesT2;      

    //status variables to reduce output of triggers in signal cases
    bool frisingGlob = false;                      //save if number of Bins over Threshold is rising or not
    bool frisingT2 = false;          
    bool frisingScaler = false;

    ushort flastnGlob = 0;                         //saves number of bins over threshold in last second
    ushort flastnT2 = 0;
    ushort flastnScaler = 0;
    int fTimeLastWrittenToT = -2*fnToTSeconds;     //initialise to a value that will not influence the first ToT; time of last ToT that was written to file

    EventCandidate flastToTTrigger;                //save last ToT, to compare n above threshold and save only at maximum
    std::vector<EventCandidate> fLTriggers;        //save L1/L2 trigger till no ToT can be formed there anymore.

    void checkLTriggers(const uint& currentSecond);//only save L-triggers in not by ToTs covered times

    //internal helper function for trigger generation
    void TriggerGenerator(const sd::LowEnergyData& data, StatusOutput& status); //method that builds triggers and saves Event Candidates

    std::pair<double, float> T2Trigger(const sd::LowEnergyData& data, StatusOutput& status, uint& nActiveT2);      //creates rsquared, s_hat (estimated average signal per station) values for T2 data ()
    std::pair<double, float> ScalerTrigger(const sd::LowEnergyData& data, StatusOutput& status, uint& nActiveScaler, double& rsquaredCropped, double& sHatCropped);

    void GenerateToT(uint& nT2, uint& nScalers, uint& nGlobal);          //a trigger based on the last fnToTSeconds s, not based on ToT trigger data!
    void CalculateChiSquare(const sd::LowEnergyData& data, uint TriggerPath, double estSignal, uint& nDof, double& chiSquare);

    void SetCandidateData(EventCandidate& ev, const sd::LowEnergyData& data); //saving current second data in the candidate
    void SetCandidateAvgs(EventCandidate& ev);                       //saving current avgs per station in the candidate

  public:
    bool foutPut = true;
    double fMaxDevFromExpChiSquare = 3;   //used for a chiSquare cut in ToT p-values: |Chi^2 - nDof|/sqrt(2*nDof) < ... (expected value over variance)
    
    Analyser(const std::string& outBase, const uint& nToTSeconds = 30);
    Analyser(const std::string& outBase, const AnalyserStatus& InitStatus, const uint& nToTSeconds = 30);
    void EndAnalysis();
    ~Analyser();

    void SetTriggerP(const double& p);                                    //parameter to declare this as EventCandidate
    void SetToTP(const double& p);                                        //parameter to declare a second as interesting for a ToT type trigger in 10s window
    void SetParameter(const unsigned& parNumber, const double& value);    //interfacing to AnalyserStatus fParameter[]
    void SetRatioToMeanVariance(const double& value);                     //for excluding outliers in variance estimation -> fMaxRatioToMeanVariance

    void AddSecondToAnalyse(const sd::LowEnergyData& data, StatusOutput& status);
    void AddVariances(float* variances);
    
    void PrintMissingStats() const;

    void Close();

    friend class DataCleaner;
    friend class DataCuts;
  };
}
#endif
