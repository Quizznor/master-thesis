#include <STAnalysis/Analyser.h>
#include <boost/math/special_functions/gamma.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
#include <utl/Accumulator.h>
#include <STAnalysis/AnalyserStatus.h>
#include <STAnalysis/StationStatus.h>
#include <interface/EventCandidate.h>
#include <interface/StatusOutPut.h>
#include <sd/LowEnergyData.h>
#include <stdexcept>

template<typename T>
T
sqr(T a)
{
  return a*a;
}

namespace ShortTerm{

  Analyser::Analyser(const std::string& outBase, const uint& nToTSeconds):
      foutEvents(outBase + "_Events.root"),
      foutBaseName(outBase),
      fmaxTriggerP(1e-5),
      fTriggerPToT(1e-3),
      fnToTSeconds(nToTSeconds),
      fpValuesGlobal(fnToTSeconds, 1),
      fpValuesScalers(fnToTSeconds, 1),
      fpValuesT2(fnToTSeconds, 1)
  {
  }

  Analyser::Analyser(const std::string& outBase, const AnalyserStatus& InitStat, const uint& nToTSeconds):
      foutEvents(outBase + "_Events.root"),
      foutBaseName(outBase),
      fStatus(InitStat),
      fmaxTriggerP(1e-5),
      fTriggerPToT(1e-3),
      fnToTSeconds(nToTSeconds),
      fpValuesGlobal(fnToTSeconds, 1),
      fpValuesScalers(fnToTSeconds, 1),
      fpValuesT2(fnToTSeconds, 1)
  {
    //fStatus.fMissingT2 = 0;
    //fStatus.fMissingScaler = 0;
  }


  Analyser::~Analyser(){
  }

  void
  Analyser::SetParameter(const unsigned& parNumber, const double& value)
  {
    fStatus.SetParameter(parNumber, value);
  }

  void 
  Analyser::SetRatioToMeanVariance(const double& value)
  {
    fMaxRatioToMeanVariance = value;
  }

  void
  Analyser::SetTriggerP(const double& p)
  {
    fmaxTriggerP = p;
  }

  void
  Analyser::SetToTP(const double& p)
  {
    fTriggerPToT = p;
  }

  void
  Analyser::Close()
  {
    foutEvents.Close(); 
  }
  

  void 
  Analyser::AddSecondToAnalyse(const sd::LowEnergyData& sData, StatusOutput& status)
  {
    if (fLastGPSsecond && fLastGPSsecond - sData.fGPSSecond > 1) {
      std::cerr << "missing GPSsecond(s)!" << std::endl;
    }
    if (fLastGPSsecond && fLastGPSsecond - sData.fGPSSecond > 3600) {
      std::cerr << "missing more than 1 hour of data, reseting!" << std::endl;
      fStatus.Reset();
    }

    TriggerGenerator(sData, status);
    fStatus.UpdateAverages(sData);
    fStatus.GetVarianceAges(status);

    fLastGPSsecond = sData.fGPSSecond;
  }


  /*
    Update the variance estimation for each station.
     Cuts are placed based on the current estimate (relativ: fMaxRatioToMeanVariance)
     or on the value itself (fStatus.fParameter[4])
  */  
  void 
  Analyser::AddVariances(float* variances)
  {
    try{
      for (uint i = 0; i < sd::kNStations; ++i)
        fStatus.fStationStatus[i].UpdateVariance(variances[i]);
    } catch (std::exception& e) {
      std::cerr << "Exception while accessing variances-array: " << e.what() << std::endl;
    }
  }

  /*
    Fill the event-Candidate with data; combine all flags into one bool
  */
  void
  Analyser::SetCandidateData(EventCandidate& ev, const sd::LowEnergyData& data)
  {
    for (uint i = 0; i < sd::kNStations; ++i) {
      ev.fData[0][i] = data.fStation[i].fTotRate;
      ev.fData[1][i] = data.fStation[i].fT2;
      ev.fData[2][i] = data.fStation[i].f70HzRate;
      ev.fData[3][i] = data.fStation[i].fT1Rate;
      ev.fData[4][i] = data.fStation[i].fScaler;

      //from ScalerTrigger
      if (fStatus[i].fActiveScaler 
          && data.fStation[i].fScaler != 0 
          && fStatus.fNProcessedTotal > 1200
          && !data.fStation[i].fJumpFlag 
          && !data.fStation[i].fUnstable
          && !data.fStation[i].fUnstableBaseline
          && !data.fStation[i].fUnstableScaler) {
        ev.fInAnalysis[1][i] = true;
      }

      //from T2Trigger
      if (fStatus[i].fActiveT2 
          && data.fStation[i].fT2 != 0 
          && !data.fStation[i].fUnstable)
        ev.fInAnalysis[0][i] = true;
    }
  }

  /*
    fill EventCandidate w.r.t the values from the analyser.fStatus:
     - averages, estimated variance
  */

  void
  Analyser::SetCandidateAvgs(EventCandidate& ev)
  {
    for (uint channel = 0; channel < 5; ++channel) {
      for (uint st = 0; st < sd::kNStations; ++st)
        ev.fAverages[channel][st] = fStatus.fStationStatus[st].fAvgs[channel].GetMean();
    }
    for (uint st = 0; st < sd::kNStations; ++st) {
      if (std::isnormal(fStatus[st].fVariance.GetMean()))
        ev.fEstimatedScalerVariance[st] = fStatus[st].fVariance.GetMean();
      else
        ev.fEstimatedScalerVariance[st] = -1;
    }
  }

  /*
    Internal function to decide wether the current second is 'interesting', i.e. a event candidate.
    Should manage the different trigger Paths and data streams (T2,scalers,...)

    Current Version:
      - 2 threshold trigger (L1 separated between Scalers and T2s):
        uses the rsquared values of T2's and Scalers to build signficances (both for them alone and a global one)
        if sign. < maxTriggerP in either T2 or Scalers a L1 called trigger is produced (encoded as trigger path 1,2 (Scalers, T2))
        if sign. of the combination of T2's and Scalers < maxTriggerP a L2 trigger is produced (enconded as 3)
      -2 ToT type triggers:
        uses a fnToTSeconds sliding window and counts the bins with p-values < TriggerPToT, again both for T2 and Scalers alone and the combination
        if more than 4 bins are above this threshold, a ToT1 or ToT2 is 'produced', where a ToT2 refers to the combination of T2 and Scalers

      Note: for combination with T2 and the ToT types a cropped version of the Scaler data is used:
            only stations with less than 1- sigma deviation from the mean are used. This reduces fake events due to single station excesses.
            However they are still put out as a L1-Scaler!

      The EventCandidates (after each of the Triggers) are temporarily saved, until it can be decided which trigger should be saved.
      In case there are more than one, the priorities are
      ToT2 > ToT1 > L2 > L1-T2 > L1-Scalers
      L triggers within the fnToTSeconds window of a ToT trigger are not saved!
      ToTs are only saved at a local maximum/end of maximum in the number of bins above threshold 
        (i.e. if the signal creates at the maximum 10 Bins above threshold, the ToT is saved at the moment when the number
        of bins above threshold decreases from 10 to 9)
    
  */
  void 
  Analyser::TriggerGenerator(const sd::LowEnergyData& data, StatusOutput& status)
  {
    //const uint NActive = fStatus.GetNActive(); //retrieve number of active station
    uint nActiveT2 = 0;
    uint nActiveScaler = 0;

    //active: has a mean and variance, that should be good
    status.fActiveT2 = fStatus.GetNActiveT2();
    status.fActiveScaler = fStatus.GetNActiveScaler();

    double rValueScalerCropped = -1;
    double pValueScalerCropped = 1;
    double sHatCroppedScaler = 0;

    //get the likelihood ratios (assuming independent variables) and the estimated signal strengths
    //the cropped value for scalers refers to taking out stations with more than 10 sigma deviation from mean first
    const std::pair<double, float> pairT2 = T2Trigger(data, status, nActiveT2);
    const double rsquaredT2 = pairT2.first;
    const std::pair<double, float> pairScaler = ScalerTrigger(data, status, nActiveScaler, rValueScalerCropped, sHatCroppedScaler);
    const double rsquaredScalers = pairScaler.first;
    
    //saving the r^2-values for the possible EventCandidate
    double sign[5] = {0, rsquaredT2, 0, 0, rsquaredScalers};

    //converting the likelihood ratio to p-Values
    double pValueGlobal = 1;
    double pValueT2 = 1;
    double pValueScalers = 1;

    if (rsquaredT2 + rsquaredScalers > 0 && rsquaredT2 > -1 && rsquaredScalers > -1) {
      pValueGlobal = boost::math::gamma_q(1, (rsquaredT2 + rValueScalerCropped)/2.);
      fStatus.fHistGlobalPValues.Fill(log10(pValueGlobal));
    }

    if (rsquaredT2 != -1) {
      pValueT2 = erfc(sqrt(rsquaredT2/2));
      fStatus.fHistT2PValues.Fill(log10(pValueT2));
    }

    if (rsquaredScalers != -1)
      pValueScalers = erfc(sqrt(rsquaredScalers/2));
    if (rValueScalerCropped > 0) {
      pValueScalerCropped = erfc(sqrt(rValueScalerCropped/2));
      fStatus.fHistScalerPValues.Fill(log10(pValueScalerCropped));
    }


    //saving the current value to the 10s window for the ToT type of trigger
    double chiSquareScaler = 0;
    uint nDofScaler = 0;
    double chiSquareT2 = 0;
    uint nDofT2 = 0;

    CalculateChiSquare(data, 2, pairT2.second, nDofT2, chiSquareT2);
    CalculateChiSquare(data, 1, sHatCroppedScaler, nDofScaler, chiSquareScaler);
    
    if (nDofT2 + nDofScaler > 0 && fabs((chiSquareScaler + chiSquareT2) - nDofScaler - nDofT2)/sqrt(2*(nDofT2 + nDofScaler)) < fMaxDevFromExpChiSquare)
      fpValuesGlobal[data.fGPSSecond % fnToTSeconds] = pValueGlobal;
    else
      fpValuesGlobal[data.fGPSSecond % fnToTSeconds] = 1;

    if (nDofScaler > 0 && fabs(chiSquareScaler - int(nDofScaler))/sqrt(2*nDofScaler) < fMaxDevFromExpChiSquare) {
      fpValuesScalers[data.fGPSSecond % fnToTSeconds] = pValueScalerCropped;
    } else {
      fpValuesScalers[data.fGPSSecond % fnToTSeconds] = 1;
      fpValuesGlobal[data.fGPSSecond % fnToTSeconds] = 1;
    }

    if (nDofT2 > 0 && fabs(chiSquareT2 - nDofT2)/sqrt(2*nDofT2) < fMaxDevFromExpChiSquare) {
      fpValuesT2[data.fGPSSecond % fnToTSeconds] = pValueT2;
    } else {
      fpValuesT2[data.fGPSSecond % fnToTSeconds] = 1;
      fpValuesGlobal[data.fGPSSecond % fnToTSeconds] = 1;
    }

    //Trigger-construction
    //different trigger paths
    bool L1_Scaler = false;
    bool L1_T2 = false;
    bool L2 = false;
    bool ToT1 = false;
    bool ToT2 = false;

    //retrieving ToT type of trigger
    uint nToTGlobal = 0;
    uint nT2 = 0;
    uint nScalers = 0;
    GenerateToT(nT2, nScalers, nToTGlobal);

    //saving 'monitoring'
    status.fnToTScaler = nScalers;
    status.fnToTT2 = nT2;
    status.fnToTGlobal = nToTGlobal;

    //Trigger - decision
    if (nT2 >= 4 || nScalers >= 4) {
      ToT1 = true;
    }
    if (nToTGlobal >= 4) {
      ToT2 = true;
    }
    if (pValueT2 < fmaxTriggerP) {
      L1_T2 = true;
    }
    if (pValueScalers < fmaxTriggerP) {
      L1_Scaler = true;
    }
    if (pValueGlobal < fmaxTriggerP) {
      L2 = true;
    }

    if((L1_Scaler || L1_T2 || L2 || ToT1 || ToT2)){
      fStatus.fTriggerDensity[data.fGPSSecond % 10000] = true;

      if(foutPut){
        if(pValueT2 < fmaxTriggerP)
          std::cout << "EventCandidate at GPSsecond " << data.fGPSSecond << " (T2) " << std::endl;
        else if(pValueScalers < fmaxTriggerP)
          std::cout << "EventCandidate at GPSsecond " << data.fGPSSecond << " (Scalers) " << std::endl;
        else
          std::cout << "EventCandidate at GPSsecond " << data.fGPSSecond << " (ToT) " << std::endl;
      }

    } else {
      fStatus.fTriggerDensity[data.fGPSSecond % 10000] = false;
    }

    //decide wether or not to put out Event candidates
    EventCandidate candidate(data.fGPSSecond, 6,  fnToTSeconds, fTriggerPToT);
    candidate.fNactiveT2 = nActiveT2;
    candidate.fNactiveScaler = nActiveScaler;

    if (rand() % 10000 == 0) {
      candidate.SetSignificances(sign);
      candidate.fEstimatedSignalT2 = pairT2.second;
      candidate.fEstimatedSignalScaler = pairScaler.second;

      SetCandidateData(candidate, data);
      SetCandidateAvgs(candidate);

      candidate.fpValue = pValueGlobal;

      candidate.fpValuesT2 = fpValuesT2;
      candidate.fpValuesScalers = fpValuesScalers;
      candidate.fpValuesGlobal = fpValuesGlobal;

      candidate.fTriggerPath = 1;
      candidate.TestUniformity();
      candidate.SaveNaboveXSigma();
      candidate.SetChiSquare();
      candidate.fTriggerPath = 6;

      foutEvents << candidate;
    }

    if ((L1_Scaler || L1_T2 || L2 || ToT1 || ToT2)) {
      candidate.fpValuesT2 = fpValuesT2;
      candidate.fpValuesScalers = fpValuesScalers;
      candidate.fpValuesGlobal = fpValuesGlobal;
    }

    //Use priority of different triggers to find which to put out (ToT also check if the number of bins is still rising)
    if(ToT2){
      double x[5] = {double(nToTGlobal), double(nT2), 0, 0, double(nScalers)};
      candidate.fTriggerPath = 5;
      candidate.SetSignificances(x);

      candidate.fpValue = candidate.GetSignificance(0);

      bool newEvent = data.fGPSSecond - flastToTTrigger.fEventTime > fnToTSeconds/2.;     //seperated and only in one second >= 4 Bins
      bool MaximumReached = nToTGlobal < flastToTTrigger.fSignificances[0] && frisingGlob;

      if ( ( newEvent ||  MaximumReached )
           && flastToTTrigger.fTriggerPath != 6                //not the one from initialisation
           && flastToTTrigger.fEventTime - fTimeLastWrittenToT > fnToTSeconds  
         ) {
        foutEvents << flastToTTrigger;
        fTimeLastWrittenToT = flastToTTrigger.fEventTime;
      }

      flastToTTrigger = candidate;
    } else if (ToT1) {
      double x[5] = { double(nToTGlobal), double(nT2), 0, 0, double(nScalers) };
      candidate.fTriggerPath = 4;
      candidate.SetSignificances(x);
      
      candidate.fpValue = std::min(candidate.GetSignificance(1), candidate.GetSignificance(4));

      bool newEvent = flastToTTrigger.fEventTime - data.fGPSSecond > fnToTSeconds/2.;
      bool maxReachedT2 = flastToTTrigger.fSignificances[1] > nT2 && frisingT2;
      bool maxReachedScaler = flastToTTrigger.fSignificances[4] > nScalers && frisingScaler;

      if( (newEvent || maxReachedT2 || maxReachedScaler)
          && flastToTTrigger.fTriggerPath != 6 
          && flastToTTrigger.fEventTime - fTimeLastWrittenToT > fnToTSeconds
          ){ 
        foutEvents << flastToTTrigger;
        fTimeLastWrittenToT = flastToTTrigger.fEventTime;
      } 

      flastToTTrigger = candidate;
    } else if (L2) {
      candidate.fTriggerPath = 3;
      candidate.SetSignificances(sign);
      candidate.fEstimatedSignalT2 = pairT2.second;
      candidate.fEstimatedSignalScaler = pairScaler.second;

      SetCandidateData(candidate, data);
      SetCandidateAvgs(candidate);

      candidate.fpValue = pValueGlobal;

      candidate.SavePull();
      candidate.TestUniformity();
      candidate.SaveNaboveXSigma();
      candidate.SetChiSquare();

      //if (candidate.fchiSquareOverDoF < 10)
        //fLTriggers.push_back(candidate);
      foutEvents << candidate;

    } else if (L1_T2) {
      candidate.fTriggerPath = 2;
      candidate.SetSignificances(sign);
      candidate.fEstimatedSignalT2 = pairT2.second;
      candidate.fEstimatedSignalScaler = pairScaler.second;

      SetCandidateData(candidate, data);
      SetCandidateAvgs(candidate);

      candidate.fpValue = pValueT2;

      candidate.SaveNaboveXSigma();
      candidate.SavePull(1);
      candidate.TestUniformity(1);
      candidate.SetChiSquare();

      //if (candidate.fchiSquareOverDoF < 10)
        //fLTriggers.push_back(candidate);
      foutEvents << candidate;

    } else if (L1_Scaler) {
      candidate.fTriggerPath = 1;
      candidate.SetSignificances(sign);
      candidate.fEstimatedSignalT2 = pairT2.second;
      candidate.fEstimatedSignalScaler = pairScaler.second;

      SetCandidateData(candidate, data);
      SetCandidateAvgs(candidate);

      candidate.fpValue = pValueScalers;

      candidate.SavePull();
      candidate.TestUniformity();
      candidate.SaveNaboveXSigma();
      candidate.SetChiSquare();

      //if (candidate.fchiSquareOverDoF < 10)   dangerous
      //fLTriggers.push_back(candidate);
      foutEvents << candidate;
    }

    //is number of Bins with p < pToT rising or not
    //for Global, T2 and Scalers seperatly
    if (nToTGlobal > flastnGlob || (frisingGlob && nToTGlobal == flastnGlob)) {
      frisingGlob = true;
    } else {
      frisingGlob = false;
    }

    if (nT2 > flastnT2 || (frisingT2 && nT2 == flastnT2)) {
      frisingT2 = true;
    } else {
      frisingT2 = false;
    }

    if (nScalers > flastnScaler || (frisingScaler && nScalers == flastnScaler)) {
      frisingScaler = true;
    } else {
      frisingScaler = false;
    }

    flastnGlob = nToTGlobal;
    flastnT2 = nT2;
    flastnScaler = nScalers;

    checkLTriggers(data.fGPSSecond);

    status.fTriggerDensity = fStatus.GetTriggerDensity();
  }//end of method

  /*
    evaluates ML of the whole array T2s with poissonian assumption
    treats the whole array as one poissonian -> L = (S+L)^N/N!*exp(-(L+S)) with L = Sum(lambda_i,i); N = Sum(N_station,station)
    returns the rsquared value, i.e. -2 * log likelihood ratio to the background only hypothesis
  */
  std::pair<double, float>
  Analyser::T2Trigger(const sd::LowEnergyData& sData, StatusOutput& status, uint& nActiveT2)
  {
    double Lambda = 0;
    double N = 0;
    uint ActiveStations = 0;

    for (uint i = 0; i < sd::kNStations; ++i) {

      if (fStatus[i].fActiveT2 
          && sData.fStation[i].fT2 != 0 
          && !sData.fStation[i].fUnstable) {

        Lambda += fStatus[i].fAvgs[1].GetMean();
        N += sData.fStation[i].fT2;
        ++ActiveStations;
      }
    }

    double rsquared = -1;

    if (ActiveStations > 0)
      rsquared = 2*(N*log(N/Lambda) - N + Lambda);
    
    //status output
    status.fMeanT2 = float(N)/ActiveStations;
    status.fnT2InAnalysis = ActiveStations;
    nActiveT2 = ActiveStations;

    return std::make_pair(rsquared, float(N - Lambda)/ActiveStations);
  }

  /*
    Similar to the T2 one, however more complex, since Scalers are not poissonian.
    Also creates a "cropped" r^2 value, where stations with more than 10 sigma deviation are excluded.
    This is used in the ToT and the combined channels

    Based on L = Prod._i 1/sqrt(2Pi sigma_i^2)*exp(-(N_i - (l_i + s))^2/(2sigma_i^2))
      results in estimator for station signal (assumed to be the same in all stations):
         s_hat = sigma^2 * Sum(N_i - Lambda_i)/sigma_i^2; with 1/sigma^2 = Sum 1/sigma_i^2
      and
        r^2 = s_hat*Sum delta_i ( with delta_i := (N_i - lambda_i)/sigma_i^2)

        Cuts before the actual analysis starts:
          - check if its active: reasonable mean, useful variance estimates (c.f. StationStatus for details)
          - Station has data that passed the cuts defined in fCuts (sets data[i].fScaler = 0)
        => computes debug output with # active stations (referring to mean and variance), mean; further cuts for Likelihood analysis
          - not marked as potentially jumping (c.f. DataCleaner.cc: MarkPotentiallyUnstableRegions())
          - not marked as unstable in T2 (c.f. DataCleaner.cc: testT2Stability())
          - not marked as having a unstable PMT baseline (c.f. DataCleaner.cc: testUnstableBaseline())
      
  */
  std::pair<double, float> 
  Analyser::ScalerTrigger(const sd::LowEnergyData& sData, StatusOutput& status, uint& nActiveScaler, double& rsquaredCropped, double& sHatCropped)
  {
    double s_hat = 0;
    double inverseVarSquared = 0;
    double Sum_deltaI = 0;
    double Lambda = 0;
    double N = 0;

    double inverseVarSquaredCropped = 0;
    double Sum_deltaICropped = 0;
    double LambdaCropped = 0;

    double ScalerTotalOut = 0;
    double meanVariance = 0;

    uint ActiveStations = 0;
    uint ActiveStationswoJumps = 0;

    for (uint i = 0; i < sd::kNStations; ++i) {
      status.fDeltaFromScalerMean[i] = -100;
      if (fStatus[i].fActiveScaler 
          && sData.fStation[i].fScaler != 0 ) {
        if ( !sData.fStation[i].fJumpFlag 
             && !sData.fStation[i].fUnstable 
             && !sData.fStation[i].fUnstableBaseline
             && !sData.fStation[i].fUnstableScaler) {

          Sum_deltaI += (sData.fStation[i].fScaler - fStatus[i].fAvgs[4].GetMean())/fStatus[i].fVariance.GetMean();
          inverseVarSquared += 1./fStatus[i].fVariance.GetMean();

          Lambda += fStatus[i].fAvgs[4].GetMean();

          N += sData.fStation[i].fScaler;
          ++ActiveStationswoJumps;
          status.fDeltaFromScalerMean[i] = (sData.fStation[i].fScaler - fStatus[i].fAvgs[4].GetMean())/sqrt(fStatus[i].fVariance.GetMean());

          if (fabs(sData.fStation[i].fScaler - fStatus[i].fAvgs[4].GetMean())/sqrt(fStatus[i].fVariance.GetMean()) < 10) {
            Sum_deltaICropped += (sData.fStation[i].fScaler - fStatus[i].fAvgs[4].GetMean())/fStatus[i].fVariance.GetMean();
            inverseVarSquaredCropped += 1./fStatus[i].fVariance.GetMean();

            LambdaCropped += fStatus[i].fAvgs[4].GetMean();
          }
        } 

        ScalerTotalOut += sData.fStation[i].fScaler;
        ++ActiveStations;
        meanVariance += fStatus[i].fVariance.GetMean();
      }
    }
    s_hat = Sum_deltaI/inverseVarSquared;
    double rsquared = -1;

    sHatCropped = Sum_deltaICropped/inverseVarSquaredCropped;
    rsquaredCropped = sHatCropped*Sum_deltaICropped;

    if (ActiveStations > 0)
      rsquared = s_hat*Sum_deltaI;
    
    //status output
    status.fMeanScaler = ScalerTotalOut/float(ActiveStations);
    status.fMeanVarianceScaler = meanVariance/ActiveStations;
    status.fnScalerInAnalysis = ActiveStationswoJumps;
    status.fMeanScalerWoJumps = N/float(ActiveStationswoJumps);

    nActiveScaler = ActiveStationswoJumps;

    return std::make_pair(rsquared, float(s_hat));
  }

  /*
    Calculates the chiSquare for a signal per Station "estSignal" in either Scaler (TriggerPath 1 ), or T2 (TriggerPath 2)
      meant for cuts based on chiSquare for ToT triggers
  */

  void 
  Analyser::CalculateChiSquare(const sd::LowEnergyData& sData, uint TriggerPath, double estSignal, uint& nDof, double& chiSquare)
  {
    if (TriggerPath == 1) {
      for (uint i = 0; i < sd::kNStations; ++i) {
        if (fStatus[i].fActiveScaler 
            && sData.fStation[i].fScaler != 0 
            && !sData.fStation[i].fJumpFlag 
            && !sData.fStation[i].fUnstable 
            && !sData.fStation[i].fUnstableBaseline
            && !sData.fStation[i].fUnstableScaler
            && fabs(sData.fStation[i].fScaler - fStatus[i].fAvgs[4].GetMean())/sqrt(fStatus[i].fVariance.GetMean()) < 10) {
          chiSquare += sqr(sData.fStation[i].fScaler - fStatus[i].fAvgs[4].GetMean() - estSignal)/(fStatus[i].fVariance.GetMean() + estSignal);
          ++nDof;
        }
      }
    } else if (TriggerPath == 2) {
      for (uint i = 0; i < sd::kNStations; ++i) {
        if (fStatus[i].fActiveT2 
            && sData.fStation[i].fT2 != 0 
            && !sData.fStation[i].fUnstable) {
          chiSquare += sqr(sData.fStation[i].fT2 - (fStatus[i].fAvgs[1].GetMean() + estSignal))/(fStatus[i].fAvgs[1].GetMean() + estSignal);
          ++nDof;
        }
      }
    }
  }

  /*
    counts the number of excess seconds in a fnToTSeconds s window and returns them

    Should be more sensitive to signals longer than a second than the
    other trigger paths 
  */
  void
  Analyser::GenerateToT(uint& nT2, uint& nScalers, uint& nGlobal)
  {
    nGlobal = 0;
    nT2 = 0;
    nScalers = 0;

    for(uint i = 0; i < fnToTSeconds; ++i){
      if(fpValuesGlobal[i] < fTriggerPToT){
        ++nGlobal;
      }
      if(fpValuesT2[i] < fTriggerPToT){
        ++nT2;
      }
      if(fpValuesScalers[i] < fTriggerPToT){
        ++nScalers;
      }
    }
  }

  void
  Analyser::EndAnalysis()
  {
    if (flastToTTrigger.fTriggerPath != 6)
      foutEvents << flastToTTrigger;

    for (const auto& ev : fLTriggers)
      foutEvents << ev;
  }

  /*
    Checks if a L-trigger is out of the time window of a potential coming ToT-trigger
      writes it to the file if so, else it will be stored until it can be decided.
  */
  void
  Analyser::checkLTriggers(const uint& currentSecond)
  {
    if (fLTriggers.size() == 0)
      return;

    bool remove[fLTriggers.size()] = { 0 };
    std::vector<EventCandidate> tmp;


    for (uint i = 0; i < fLTriggers.size(); ++i) {
      double evtTime = fLTriggers[i].fEventTime;

      if (evtTime < currentSecond - fnToTSeconds) { //no ToT1/2 for this time -> put out Event (and remove from list)
        foutEvents << fLTriggers[i];
        remove[i] = true;
      } else if (flastToTTrigger.fEventTime - fnToTSeconds <= evtTime && evtTime <= flastToTTrigger.fEventTime) {  //
        remove[i] = true;
      }
    }

    for (uint i = 0; i < fLTriggers.size(); ++i) {
      if (!remove[i])
        tmp.push_back(fLTriggers[i]);
    }

    fLTriggers = tmp;
  }

  void
  Analyser::PrintMissingStats() const
  {
    std::cout << "Missing T2 seconds: " << fStatus.fMissingT2 << " out of " << fStatus.fNProcessedTotal << std::endl;
    std::cout << "Missing scaler seconds: " << fStatus.fMissingScaler << " out of " << fStatus.fNProcessedTotal << std::endl;
  }
}
