#include <sd/Constants.h>
#include <sd/LowEnergyData.h>
#include <interface/circ_array.h>
#include <interface/StatusOutPut.h>
#include <io/RootOutFile.h>
#include <io/RootInFile.h>
#include <string>
#include <utl/Accumulator.h>
#include <interface/JumpOutput.h>
#include <TH1F.h>
#include <TH2F.h>

#ifndef _DataCleaner_
#define _DataCleaner_

struct StatusOutput;
class AverageBuilder;

typedef unsigned int uint;
typedef unsigned short ushort;

/**
 * Main Class to manage the Cuts and instability flags of the merged-1s-based data
 *  helper classes for different cuts 
 */

namespace ShortTerm{

  class Analyser;
  class DataCleaner;

  //variables bundeled for searching for jumps with a linear fit based algorithm
  //saves the Chi^2 values and the Coefficients of the fit, while scanning the interval
  struct JumpSearchVar{
    std::vector<double> fChi1;
    std::vector<double> fChi2;

    utl::Accumulator::LinearFitChi2 fFit1;
    utl::Accumulator::LinearFitChi2 fFit2;

    std::vector<double> fCoeff0_1;
    std::vector<double> fCoeff0_2;

    std::vector<double> fCoeff1_1;
    std::vector<double> fCoeff1_2;

    std::vector<double> fChiValues;             //sum of the Chi1, Chi2

    double fChiInit;
    int fcpCandidate;
  };


  //Values for the cuts on scaler data (except one for T2)
  struct DataCuts{
    ushort fMaxValueScaler = 10000;             // -> use as maximal value for a 'stable' station
    ushort fMinValueScaler = 1000;              //not in use in the current version
    ushort fMaxVariance = 10000;                //used in detection of unstable Scaler stations
    float fMaxVarianceInterval = 100000;         

    float fVarianceBaselineCut = 2;             //see Baseline-Variance histogram

    ushort fMaxValueT2 = 1000;                   

    float fMinAoP = 2;                          
    float fMaxAoP = 5;                          

    float fMaxBaselineValue = 200;              

    float fminDistanceToInt = 0.031;            //resolution of monitoring data is 0.01; 0.031 also excludes 0.03.

    float fMaxScalerDevFromAvg = 10;            //maximal deviation from Avg in Scalers, i.e. |(N_i - Lambda_i)/sigma_i| < fMaxScalerDevFromAvg;
                                                // only for cropped versions of analysis, and for protecting the mean values from outliers.

    const float fTubeMask1 = 7;                 //allowed value -> 3 working PMTs
    //const float fTubeMask2 = 15;                //allowed value -> 3 working PMTs

    bool CloseToIntValue(float* Baselines);
    void ApplyCuts(sd::LowEnergyData& data, StatusOutput& status);//, DataCleaner* cleaner);
  };

  class DataCleaner
  {
  private:
    io::RootOutFile<StatusOutput> foutStatus;

    TH1F fHistBaselineVar;
    TH1F fHistVar;
    TH2F fT2VarHist;
    TH2F fScalerMeanVar;

    circ_array<5000, sd::LowEnergyData> fBuffer;

    //JumpOutput fJumpOut;
    uint ffirstGPSsecond = 0;

    Analyser* fAnalyser;
    AverageBuilder* fAvger;

    const uint fIntervalLength = 120;        //for the fit/variance based detection of jumps (which is not used at the moment), also for scaler instability test
    const double fThreshold  = 3;           //for variance based exclusion of unstable data: Jumps in scalers (c.f. above, not in use); for the T2 instability exclusion

    const uint fCutJumpBefore = 300;
    const uint fCutJumpAfter = 300;

    void SearchForPotentiallyUnstableRegions();
    void JumpSearch();
    void TestT2Stability();
    void TestBaselineStability();
    void TestScalerStability();
    void GetIntervalVariances(const uint& i_first, const uint& i_last, 
                              utl::Accumulator::Var* avgVariances, utl::Accumulator::Mean* meanScalerRate, 
                              uint* nInVariance);  
                                                //for instability test of scalers. Avoids iterating over Jumps and artificially increasing variance
    void FitSlices(const uint& i_first, const uint& i_last);  //helper method for jump detection (searches in one single interval)

    void MarkJump(const uint& Id, const uint& SecondInBuffer);
    void MarkUnstable(const uint& Id);                 //Based on T2-data. uses JumpFlag as well, marks whole Buffer
    void MarkUnstableBaseline(const uint& Id);         //Based on the variance of the Baseline of all PMTs, if one is above 1, this station is removed
    void MarkUnstableScalers(const uint& Id);          //Based on Scaler variance uses fUnstableScaler flag
    void MarkUnstableScalers(const uint& Id, const uint& secondInBuffer); //same as above, but on fIntervalLength level
    void PushOutSecond();                       //gives data to analyser, avger and creates Status Output

  public:
    DataCuts fCut;

    DataCleaner(const std::string& outBase);
    DataCleaner(const std::string& outBase, const uint& IntervalLength, const double& Threshold, const uint& JumpCutBefore, const uint& JumpCutAfter);
    ~DataCleaner() {}

    void AddData(const sd::LowEnergyData& data);
    //void ScanData(io::RootInFile<sd::LowEnergyData>& Data, uint NSeconds); 
    void EndAnalysis();

    void SetAnalyser(Analyser& a) { fAnalyser = &a; }
    void SetAvger(AverageBuilder& b) { fAvger = &b; }

    friend class DataCuts;
  };
}
#endif
