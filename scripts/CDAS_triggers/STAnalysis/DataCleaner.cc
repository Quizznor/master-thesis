#include <STAnalysis/DataCleaner.h>
#include <STAnalysis/Analyser.h>
#include <STAnalysis/AnalyserStatus.h>
#include <STAnalysis/StationStatus.h>
#include <interface/AverageBuilder.h>
#include <interface/StatusOutPut.h>
#include <algorithm>
#include <cmath>
#include <utility>
#include <iostream>
#include <limits>


namespace ShortTerm{
  void
  DataCuts::ApplyCuts(sd::LowEnergyData& data, StatusOutput& status)
  {
    for (uint i = 0; i < sd::kNStations; ++i) {
      
      //statistics 
      if (data[i].fScaler != 0)
        ++status.fnScalerWithData;
      if (data[i].fT2 != 0)
        ++status.fnT2WithData;

      //Applying cuts and setting the scaler rate to zero if outside window
      // also sets the counters how many stations were cut by each parameter
      if (!(data[i].fTubeMask == fTubeMask1) ) {
        if (data[i].fScaler) //only count stations with data
          ++status.fCutTubeMask;
        data[i].fScaler = 0;
      } else if (data[i].fAreaOverPeak > fMaxAoP || data[i].fAreaOverPeak < fMinAoP) {
        if (data[i].fScaler) //only count stations with data
          ++status.fCutAoP;
        data[i].fScaler = 0;
      } else if (data[i].fScaler == std::numeric_limits<unsigned short>::max()) {//from failed conversion of line to ScalerRow.h 
        ++status.fCutScalerRate;
        data[i].fScaler = 0;        
      } else if (data[i].fScaler > 2*fMaxValueScaler) { //fMaxValueScaler is adjusted to mean values...so adding a factor two for second based data
        ++status.fCutScalerRate;
        data[i].fScaler = 0;
      }

      //Cut on Baseline from Monitoring: if there is no monitoring data - this station will not be used
      // because then the 'jump'-detection does not work
      bool noBaselineMonit = false;
      for (uint j = 0; j < 3; ++j) {
        if (data[i].fPMTBaseline[j] <= 0 || data[i].fPMTBaseline[j] > fMaxBaselineValue) {
          data[i].fPMTBaseline[j] = 0;
          noBaselineMonit = true;
        }
      }
      if (noBaselineMonit) {
        if (data[i].fScaler)
          ++status.fCutBaseline;
        data[i].fScaler = 0;
      }

      //Cut on crazy values of T2's
      if (data[i].fT2 > fMaxValueT2) {
        data[i].fT2 = 0;
        ++status.fCutT2;
      }

      //collecting Information about the instability flags
      if (data[i].fJumpFlag)
        status.fJumpIds.push_back(i);
      if (data[i].fUnstable)
        status.fUnstableIds.push_back(i);
      if (data[i].fUnstableBaseline)
        status.fUnstableBaseline.push_back(i);
      if (data[i].fUnstableScaler)
        status.fUnstableScaler.push_back(i);
    }
  }

  //Return true if value is close to integer crossing
  bool
  DataCuts::CloseToIntValue(float* Baselines)
  {
    const float min = fminDistanceToInt;
    const float max = 1 - min;

    for (int i = 0; i < 3; ++i) {
      if (fabs(Baselines[i] - int(Baselines[i])) < min || fabs(Baselines[i] - int(Baselines[i])) > max) {
        if (Baselines[i] > 0.001)
          return true;
        else
          return false;   //if a baseline is 0 the scaler data is set to zero anyway -> not marking!
      }
    }
    return false;
  }


  DataCleaner::DataCleaner(const std::string& basename) : 
    foutStatus(basename + "_Status.root"), 
    fHistBaselineVar("bv", "bv", 5000, 0, 50),
    fHistVar("v", "v", 500, 0, 50000),
    fT2VarHist("T2VarHist","T2VarHist", 1000, 0, 100, 1000, 0, 100),
    fScalerMeanVar("ScalerMeanVar","ScalerMeanVar", 2000, 0, 20000, 500, 0, 50000)
    {}

  DataCleaner::DataCleaner(const std::string& basename, 
                           const uint& IntervalLength, 
                           const double& Threshold,
                           const uint& CutJumpBefore = 300,
                           const uint& CutJumpAfter = 300)
     : foutStatus(basename + "_Status.root"),
      fHistBaselineVar("bv", "bv", 5000, 0, 50),
       fHistVar("v", "v", 500, 0, 500000),
       fT2VarHist("T2VarHist","T2VarHist", 1000, 0, 100, 1000, 0, 100),
       fScalerMeanVar("ScalerMeanVar","ScalerMeanVar", 2000, 0, 20000, 500, 0, 50000),
       fIntervalLength(IntervalLength),
       fThreshold(Threshold),
       fCutJumpBefore(CutJumpBefore),
       fCutJumpAfter(CutJumpAfter)

  {}

  /*
    Manages the buffer and the various instability detections.
     If the buffer is full the algorithms are started and the appropriate number of 
     seconds is forwarded to the analysis part
  */
  void
  DataCleaner::AddData(const sd::LowEnergyData& sd) 
  {
    fBuffer.push_back(sd);

    if (fBuffer.GetN() == fBuffer.size()) {    
      JumpSearch();

      while(fBuffer.GetN() > fCutJumpAfter + fCutJumpBefore)
        PushOutSecond();
    } 
  }

  void
  DataCleaner::EndAnalysis()
  {
    JumpSearch();

    while (!fBuffer.empty())
      PushOutSecond();

    fAnalyser->EndAnalysis();

    foutStatus.Write(fAnalyser->fStatus);
    foutStatus.Write(fHistBaselineVar);
    foutStatus.Write(fHistVar);
    foutStatus.Write(fT2VarHist);
    foutStatus.Write(fScalerMeanVar);
  }

  void
  DataCleaner::PushOutSecond()
  {
    auto le = fBuffer.pop_front();

    StatusOutput stat(le.fGPSSecond);

    fCut.ApplyCuts(le, stat);

    //Avg before Analyser, so in the case that no Variance is set, 
    // this is done before the Analyser starts
    fAvger->AddData(le);
    fAnalyser->AddSecondToAnalyse(le, stat);

    foutStatus << stat;
  }



  /*
    method that coordinates search for Jumps
      actual jump detection is done in FitSlices (for algorithm based approaches)
  */
  void
  DataCleaner::JumpSearch()
  {
    TestT2Stability();                         //variance of T2's incompatible with a Poissonian at fTreshold level
    TestBaselineStability();
    SearchForPotentiallyUnstableRegions();     //Based on the current baselines differences to int values
    TestScalerStability();

    /*if (fBuffer.GetN() == fBuffer.size()) {
      for (uint i = 0; i <= fBuffer.size() - fIntervalLength - fCutJumpAfter; i += fIntervalLength/2.) 
        this->FitSlices(i, i + fIntervalLength);
    } else if (fBuffer.GetN() > fIntervalLength) {
      for (uint i = 0; i < fBuffer.GetN() - fIntervalLength; i += fIntervalLength/2.) 
        this->FitSlices(i, i + fIntervalLength);

      this->FitSlices(fBuffer.GetN() - fIntervalLength - 1, fBuffer.GetN() - 1);
    } else {
      this->FitSlices(0, fBuffer.GetN());
    }*/
  }


  /* 
    Use the assumption that T2's are poissonians and that unstable stations show
     a deviation in variance from this 'baseline'

     if the estimated variance exceeds the expected one (sqrt(mean)) by fThreshold sigmas,
      which are calculated using the variance of the variance estimator, the station is excluded for all channels
      -> Unstable-Flag
  */
  void
  DataCleaner::TestT2Stability()
  {
    utl::Accumulator::Var varT2[sd::kNStations];

    for (const auto& le : fBuffer) {
      for (uint i = 0; i < sd::kNStations; ++i) {
        if (le.fStation[i].fT2 != 0)
          varT2[i](le.fStation[i].fT2);
      }
    }

    for (uint i = 0; i < sd::kNStations; ++i) {
      double expected_varVariance = 2*varT2[i].GetMean()*varT2[i].GetMean();    //use the mean + poissonian hypothesis as expectation
      expected_varVariance /= (varT2[i].GetCount() - 1);

      fT2VarHist.Fill(varT2[i].GetMean(), varT2[i].GetVar());

      if (varT2[i].GetCount() > 0)
        if ( (varT2[i].GetVar() - varT2[i].GetMean())/sqrt(expected_varVariance) > fThreshold )  
          MarkUnstable(i);
    }
    
  }

  /*
    Check the variance of the baseline to find stations with unstable baselines
     -> cut on absolute value of the variance (note that the variance is smaller than in reality, due to the repetition of 
      values from monitoring)
       Cut value is defined in fCut.fVarianceBaselineCut

    Fills the UnstableBaseline flags
  */

  void
  DataCleaner::TestBaselineStability()
  {
    utl::Accumulator::Var varBaseline[sd::kNStations][3];

    for (const auto& le : fBuffer) {
      for (uint i = 0; i < sd::kNStations; ++i) {
        for (uint j = 0; j < 3; ++j) 
          if (le.fStation[i].fScaler)
            varBaseline[i][j](le.fStation[i].fPMTBaseline[j]);
      }
    }

    for (uint i = 0; i < sd::kNStations; ++i) {
      bool unstableBaseline = false;

      for (int j = 0; j < 3; ++j) {
        if (varBaseline[i][j].GetMean() != 0 && varBaseline[i][j].GetVar() > fCut.fVarianceBaselineCut)
          unstableBaseline = true;
        
        if (varBaseline[i][j].GetMean() > 0 && varBaseline[i][j].GetMean() < 100)
          fHistBaselineVar.Fill(varBaseline[i][j].GetVar()); 
      }

      if (unstableBaseline)
        MarkUnstableBaseline(i);
    }
  }

  /*
    Searches for PMT-Baselines close to int values.
     It is expected to have jumps in the rate for these values.
     Uses fCut.fminDistanceToInt in fCut.CloseToIntValue() to find those stations
     -> fills the JumpFlag
  */
  void
  DataCleaner::SearchForPotentiallyUnstableRegions()
  {
    auto itEnd = fBuffer.end();
    itEnd -= fCutJumpAfter;

    for (auto it = fBuffer.begin(); it != itEnd && it != fBuffer.end(); it += 30) {
      for (uint i = 0; i < sd::kNStations; ++i) {
        if (fCut.CloseToIntValue((*it)[i].fPMTBaseline)) {
          const uint secondInBuffer = it->fGPSSecond - fBuffer.begin()->fGPSSecond;
          MarkJump(i, secondInBuffer);
        }
      }
    }
  }

  void
  DataCleaner::TestScalerStability()
  {
    utl::Accumulator::Var varScaler[sd::kNStations];
    utl::Accumulator::Mean meanScaler[sd::kNStations];
    uint nInVariance[sd::kNStations] = { 0 };

    for (uint i = 0; i < fBuffer.size() - (fCutJumpAfter + fIntervalLength); i += fIntervalLength) {
      GetIntervalVariances(i, i + fIntervalLength, varScaler, meanScaler, nInVariance);
    }

    for (uint i = 0; i < sd::kNStations; ++i) {
      if (varScaler[i].GetCount() < 5 || !varScaler[i].GetMean())
        continue;
      fScalerMeanVar.Fill(meanScaler[i].GetMean(), varScaler[i].GetMean());

      if (std::isnormal(varScaler[i].GetMean()) && varScaler[i].GetMean() > fCut.fMaxVariance) {
        MarkUnstableScalers(i);
      } else if (std::isnormal(meanScaler[i].GetMean()) && meanScaler[i].GetMean() > fCut.fMaxValueScaler) {
        MarkUnstableScalers(i);
      }
    }
  }

  void
  DataCleaner::GetIntervalVariances(const uint& i_first,
                                    const uint& i_last, 
                                    utl::Accumulator::Var* avgVariances, 
                                    utl::Accumulator::Mean* meanScalerRate, 
                                    uint* nInVariance)
  {
    utl::Accumulator::Var var[sd::kNStations];
    bool Flagged[sd::kNStations] = { false }; 

    for (uint i = i_first; i < i_last; ++i) {
      for (uint j = 0; j < sd::kNStations; ++j) {
        uint tmp = fBuffer[i][j].fScaler;

        if (Flagged[j] || tmp == std::numeric_limits<unsigned short>::max())
          continue;

        if (var[j].GetCount() > 2) {
          if (tmp && tmp < var[j].GetMean() + 5000 ) {
            var[j](tmp);
          }
        } else {
          if (tmp != 0 && tmp < fCut.fMaxValueScaler && tmp > fCut.fMinValueScaler) {
            var[j](tmp);
          }
        }
        if (fBuffer[i][j].fJumpFlag) {
          Flagged[j] = true;
        }
      }
    }

    for (uint i = 0; i < sd::kNStations; ++i) {
      if (Flagged[i])
        continue;
      if (std::isnormal(var[i].GetVar())) {
        fHistVar.Fill(var[i].GetVar());
        avgVariances[i](var[i].GetVar());
        nInVariance[i] += var[i].GetCount();
        meanScalerRate[i](var[i].GetMean());

        if (var[i].GetVar() > fCut.fMaxVarianceInterval)
          MarkUnstableScalers(i, i_first + fIntervalLength/2.);
      }
    }
  }


  /*
    Helper methods that mark the data in the buffer around found instabilities
  */
  void
  DataCleaner::MarkUnstableBaseline(const uint& Id)
  {
    auto itEnd = fBuffer.end();
    itEnd -= fCutJumpAfter + fCutJumpBefore;  //this is what remains in the buffer for the next check, so don't mark it 'twice'

    for (auto it = fBuffer.begin(); it != itEnd; ++it)
      (*it)[Id].fUnstableBaseline = true;
  }

  void
  DataCleaner::MarkUnstableScalers(const uint& Id)
  {
    auto itEnd = fBuffer.end();
    itEnd -= fCutJumpAfter + fCutJumpBefore;  //this is what remains in the buffer for the next check, so don't mark it 'twice'

    for (auto it = fBuffer.begin(); it != itEnd; ++it)
      (*it)[Id].fUnstableScaler = true;
  }

  void
  DataCleaner::MarkUnstableScalers(const uint& Id, const uint& secondInBuffer)
  {
    auto it1 = fBuffer.begin();
    if (secondInBuffer > fCutJumpBefore)
      it1 += secondInBuffer - fCutJumpBefore;

    auto itEnd = fBuffer.begin();
    itEnd += secondInBuffer + fCutJumpAfter + 1;

    for (; it1 != itEnd; ++it1)
      (*it1)[Id].fUnstableScaler = true;
  }

  void
  DataCleaner::MarkUnstable(const uint& Id)
  {
    auto itEnd = fBuffer.end();
    itEnd -= fCutJumpAfter + fCutJumpBefore;  //this is what remains in the buffer for the next check, so don't mark it 'twice'

    for (auto it = fBuffer.begin(); it != itEnd; ++it)
      (*it)[Id].fUnstable = true;
  }

  void
  DataCleaner::MarkJump(const uint& Id, const uint& secondInBuffer)
  {
    auto it1 = fBuffer.begin();
    if (secondInBuffer > fCutJumpBefore)
      it1 += secondInBuffer - fCutJumpBefore;
    auto itEnd = fBuffer.begin();
    itEnd += secondInBuffer + fCutJumpAfter + 1;

    for (; it1 != itEnd; ++it1)
      (*it1)[Id].fJumpFlag = true;
  }


  /*
    Old: (as a comment below)
    Fits two linear functions to scaler data and compares them to a single one
      (c.f. FindJumps.cxx)
    Works on the whole array at once (maybe better in terms of cache efficiency)
    - added a cut: requiring a minimal difference at CP from the fits of at least 25 (hard coded)

    new version:
      Works with the variance of a 1 min (= default value, else it's Intervallength) mean. 
      Detection based on Distribution of a estimated Variance (variance of the variance-estimator) 
      for Gaussian distributions: Var[Var] = 2 sigma^4/(n - 1)

      Cut away of current estimate deviates more then fThreshsold stds from the mean (estimated in fAnalyser->fStatus[ ])
      default is 2.5 -> Tau(fp) apporx 5000 s
  */
  void
  DataCleaner::FitSlices(const uint& i_first, const uint& i_last)
  {
    utl::Accumulator::Var var[sd::kNStations];

    for (uint i = i_first; i < i_last; ++i) {
      for (uint j = 0; j < sd::kNStations; ++j) {
        uint tmp = fBuffer[i][j].fScaler;

        if (var[j].GetCount() > 0) {
          if (tmp != 0 
              && (fBuffer[i][j].fTubeMask == 7 || fBuffer[i][j].fTubeMask == 15)
              && tmp < var[j].GetMean() + 700 
              && tmp > var[j].GetMean() - 700) {
            var[j](tmp);
          }
        } else {
          if (tmp != 0 && tmp < fCut.fMaxValueScaler && tmp > fCut.fMinValueScaler) {
            var[j](tmp);
          }
        }
      }
    }

    for (uint i = 0; i < sd::kNStations; ++i) {
      float tmp = var[i].GetVar();

      float varVariance = 0;
      float scaledDeviation = 0;

      if (std::isnormal(tmp) && fAnalyser->fStatus[i].fVariance.GetCount() > 10) {
        varVariance = fAnalyser->fStatus[i].fVariance.GetMean();
        varVariance *= varVariance;
        varVariance *= 2/(var[i].GetCount() - 1);

        scaledDeviation = tmp - fAnalyser->fStatus[i].fVariance.GetMean();
        scaledDeviation /= sqrt(varVariance);
      }

      if (varVariance) {
        if (scaledDeviation > fThreshold)
          MarkJump(i, i_first + fIntervalLength/2.);
      } else {
        if (std::isnormal(tmp) && tmp > 4500)
          MarkJump(i, i_first + fIntervalLength/2.);
      }
    }
  }
}

/*
  various code snippets, that can be used if a detailed jump detection is needed
*/
//old code: Jump search based on linear fits, was replaced by variance based algorithm (also not in use)

/*if ( (i_last - i_first) % 4 != 0)
      std::cerr << "Warning: Intervalsize does not match stepsize of fitting algorithm! (" << i_first << "," << i_last << ")" << std::endl;

    JumpSearchVar FitVar[sd::kNStations];

    for (uint i = i_first; i < i_last; ++i) {
      //forward fit
      for (uint j = 0; j < sd::kNStations; ++j){
        uint tmp = fBuffer[i][j].fScaler;
        if (tmp != 0 && tmp < fCut.fMaxValueScaler && tmp > fCut.fMinValueScaler) {
          FitVar[j].fFit1(i - i_first, tmp, 50);
        }
      }
      //backward fit (seperate loop, to make it more cache efficient (?))
      for (uint j = 0; j < sd::kNStations; ++j) {
        uint tmp = fBuffer[i_last - i - 1 + i_first][j].fScaler;
        if (tmp != 0 && tmp < fCut.fMaxValueScaler && tmp > fCut.fMinValueScaler) {
          FitVar[j].fFit2(i_last - 1 - i, tmp, 50);
        }
      }

      //saving values of Chi^2
      if ((i - i_first) && i % 4 == 0) {
        for (uint j = 0; j < sd::kNStations; ++j){
          float tmp = FitVar[j].fFit1.GetChi2();
          if (std::isnormal(tmp)) {
            FitVar[j].fChi1.push_back(tmp);  
            FitVar[j].fCoeff0_1.push_back(FitVar[j].fFit1.GetCoefficients().first);
            FitVar[j].fCoeff1_1.push_back(FitVar[j].fFit1.GetCoefficients().second);
          } else {
            FitVar[j].fChi1.push_back(1e6);  
            FitVar[j].fCoeff0_1.push_back(0);
            FitVar[j].fCoeff1_1.push_back(0);
          }            
        }        
      } else if ((i - i_first - 2) % 4 == 0) {
        for (uint j = 0; j < sd::kNStations; ++j){
          float tmp = FitVar[j].fFit1.GetChi2();
          if (std::isnormal(tmp)) {
            FitVar[j].fChi2.push_back(tmp);
            FitVar[j].fCoeff0_2.push_back(FitVar[j].fFit2.GetCoefficients().first);
            FitVar[j].fCoeff1_2.push_back(FitVar[j].fFit2.GetCoefficients().second);  
          } else {
            FitVar[j].fChi2.push_back(1e6);
            FitVar[j].fCoeff0_2.push_back(0);
            FitVar[j].fCoeff1_2.push_back(0);  
          }           
        }
      }
    }

    //Building the Chi^2 vectors
    for (uint i = 0; i < sd::kNStations; ++i) 
      if (FitVar[i].fChi2.size() - 1 == FitVar[i].fChi1.size()) 
        for (uint j = 0; j < FitVar[i].fChi1.size(); ++j) 
          FitVar[i].fChiValues.push_back( (FitVar[i].fChi1[j] + FitVar[i].fChi2[FitVar[i].fChi2.size() - j - 2])/(FitVar[i].fFit1.GetNdof() - 2) );

    //finding best Fit (2 fits), saving Chi^2 for one single fit
    for (uint i = 0; i < sd::kNStations; ++i) {
      float tmp = FitVar[i].fFit1.GetChi2()/FitVar[i].fFit1.GetNdof();
      if (std::isnormal(tmp)) {
        FitVar[i].fChiInit = tmp;
        FitVar[i].fcpCandidate = min_element(FitVar[i].fChiValues.begin(), FitVar[i].fChiValues.end()) - FitVar[i].fChiValues.begin();
      } else {
        FitVar[i].fChiInit = 0;   //then no cp can be found
        FitVar[i].fcpCandidate = 0;
      }      
    }

    //comparing Chi^2 values and possibly marking Jumps
    for (uint i = 0; i < sd::kNStations; ++i) {
      if (FitVar[i].fChiValues.size() > 0) {
        float delta = (FitVar[i].fCoeff0_1[FitVar[i].fcpCandidate] - FitVar[i].fCoeff0_2[FitVar[i].fCoeff0_2.size() - FitVar[i].fcpCandidate - 2]);
        delta += (FitVar[i].fCoeff1_1[FitVar[i].fcpCandidate]
                  - FitVar[i].fCoeff1_2[FitVar[i].fCoeff0_2.size() - FitVar[i].fcpCandidate - 2]) 
                  * 4*(FitVar[i].fcpCandidate + 1) ;

        if ( FitVar[i].fChiValues[FitVar[i].fcpCandidate] < (1 - fThreshold)*FitVar[i].fChiInit 
             && fabs(delta) > fMinDeltaJump) 
          this->MarkJump(i, 4*FitVar[i].fcpCandidate + i_first + 4);          
      }
    }*/

/*
  // old code: searches for the right GPSsecond % 61 = x for the jump search based on mean values with 61s intervals

  void
  DataCleaner::ScanData(io::RootInFile<sd::LowEnergyData>& Data, uint NSeconds)
  {
    double chiValues[61][sd::kNStations] = { { 0 } };
    utl::Accumulator::Var variances[61][sd::kNStations];

    for (const auto& ev : Data) {
      if (!ffirstGPSsecond)
        ffirstGPSsecond = ev.fGPSSecond;

      if ((ev.fGPSSecond - ffirstGPSsecond) % 10000 == 0)
        std::cout << "processing second " << ev.fGPSSecond - ffirstGPSsecond << std::endl;

      if ((ev.fGPSSecond - ffirstGPSsecond) >= NSeconds)
        break;

      for (ushort modulo = 0; modulo < 61; ++modulo) {
        for (uint i = 0; i < sd::kNStations; ++i) {
          if (ev.fGPSSecond % 61 != modulo) {
            if (ev.fStation[i].fScaler != 0)
              variances[modulo][i](ev.fStation[i].fScaler);
          } else {
            if (variances[modulo][i].GetN() > 2) {
              chiValues[modulo][i] += variances[modulo][i].GetVar()*variances[modulo][i].GetN();
              variances[modulo][i].Clear();
            }
            variances[modulo][i].Clear();
            if (ev.fStation[i].fScaler != 0)
              variances[modulo][i](ev.fStation[i].fScaler);           
          }
        }
      }
    }

    double tmpMin[sd::kNStations];
    for (uint modulo = 0; modulo < 61; ++modulo) {
      for (uint i = 0; i < sd::kNStations; ++i) {
        if (i == 450)
          std::cout << "tmpMin[i]: " << tmpMin[i] << " for modulo: " << modulo << " chiValues[i]: " << chiValues[modulo][i] << std::endl; 
        if (!modulo)
          tmpMin[i] = chiValues[modulo][i];
        else if (tmpMin[i] > chiValues[modulo][i]) {
          fJumpOut.fModuloStation[i] = modulo;
          tmpMin[i] = chiValues[modulo][i];
        }
      }
    }
  }*/


//old code: Jump Search based on 61s intervals
  /*void
  DataCleaner::Get61sJumps()
  {
    //temporary storing the values that will be used to search for jumps
    std::vector<std::pair<double , double> > meanVariances[sd::kNStations];   //saving (mean, var/N)
    utl::Accumulator::Var variance[sd::kNStations];

    uint firstGPSsecond = 0;

    for (const auto& le : fBuffer) {
      if (!firstGPSsecond)
        firstGPSsecond = le.fGPSSecond;
      for (uint i = 0; i < sd::kNStations; ++i) {

        if (le.fGPSSecond % 61 != fJumpOut.fModuloStation[i]) {
          if (le.fStation[i].fScaler != 0 && le.fStation[i].fScaler < 4000) //Quality Cuts
            variance[i](le.fStation[i].fScaler);
        } else {
          if (variance[i].GetN() > 2) {
            double mean = variance[i].GetMean();
            double var = variance[i].GetVar();
            int N = variance[i].GetN();

            meanVariances[i].push_back(std::make_pair(mean, var/N));
          }

          variance[i].Clear();
          if (le.fStation[i].fScaler != 0 && le.fStation[i].fScaler < 4000) //Quality Cuts
            variance[i](le.fStation[i].fScaler);
        }
      }
    }

    //finding the jumps
    for (uint i = 0; i < sd::kNStations; ++i) {
      if (meanVariances[i].size() < 2)
        continue;
      for (uint j = 0; j < meanVariances[i].size() - 1; ++j) {
        float figureOfMerrit = fabs(meanVariances[i][j].first - meanVariances[i][j + 1].first);
        figureOfMerrit /= sqrt(meanVariances[i][j].second + meanVariances[i][j + 1].second);

        if (figureOfMerrit > fMerritThreshold) {
          fJumpOut.fJumpTimes[i].push_back(j*61 + firstGPSsecond + fJumpOut.fModuloStation[i]);
          fJumpOut.fJumpHeights[i].push_back(fabs(meanVariances[i][j].first - meanVariances[i][j + 1].first));
          fJumpOut.fMerrit[i].push_back(figureOfMerrit);
        }
      }
    }

  }*/
