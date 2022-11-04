#include <fwk/CentralConfig.h>
#include <fwk/RunController.h>
#include <fwk/SVNGlobalRevision.h>

#include <det/Detector.h>
#include <sdet/SDetector.h>
#include <sdet/Station.h>
#include <sdet/StationTriggerAlgorithm.h>
#include <sdet/PMTConstants.h>

#include <evt/Event.h>

#include <sevt/EventTrigger.h>
#include <sevt/Header.h>
#include <sevt/SEvent.h>
#include <sevt/Station.h>
#include <sevt/Scintillator.h>
#include <sevt/PMT.h>
#include <sevt/PMTCalibData.h>
#include <sevt/SmallPMT.h>
#include <sevt/SmallPMTCalibData.h>
#include <sevt/PMTRecData.h>
#include <sevt/StationGPSData.h>
#include <sevt/StationRecData.h>
#include <sevt/ScintillatorRecData.h>
#include <sevt/StationTriggerData.h>
#include <sevt/StationCalibData.h>
#include <sevt/SignalSegment.h>

#include <utl/TimeStamp.h>
#include <utl/TimeInterval.h>
#include <utl/Trace.h>
#include <utl/TraceAlgorithm.h>
#include <utl/Reader.h>
#include <utl/ErrorLogger.h>
#include <utl/Math.h>
#include <utl/Accumulator.h>
#include <utl/QuadraticFitter.h>
#include <utl/ExponentialFitter.h>
#include <utl/String.h>
#include <utl/ShadowPtr.h>
#include <utl/TabularStream.h>

#include "SdCalibrator.h"
#include "CalibrationParameters.h"

#include <iostream>
#include <sstream>
#include <algorithm>

#define DO_PEAK_HISTOS 1

#define USE_SPMT_SIGNAL_AS_TOTAL 0

#include <config.h>

using namespace fwk;
using namespace evt;
using namespace sevt;
using namespace utl;
using namespace std;


namespace SdCalibratorOG {

  class UnrestrictedPMTRecData : public PMTRecData {
  public:
    void RemoveVEMTrace() { fTrace.RemoveTrace(StationConstants::eTotal); }
  };


  // pair<> output helper
  template<typename T1, typename T2>
  inline ostream&
  operator<<(ostream& os, pair<T1, T2>& p)
  {
    return os << '[' << p.first << ", " << p.second << ']';
  }


  // Read parameters from XML files
  VModule::ResultFlag
  SdCalibrator::Init()
  {
    Branch topB =
      CentralConfig::GetInstance()->GetTopBranch("SdCalibrator");

    topB.GetChild("PMTSummationCutoff").GetData(fPMTSummationCutoff);
    topB.GetChild("validDynodeAnodeRatioRange").GetData(fValidDynodeAnodeRatioRange);
    topB.GetChild("peakFitRange").GetData(fPeakFitRange);
    topB.GetChild("peakFitChi2Accept").GetData(fPeakFitChi2Accept);
    topB.GetChild("peakVEMConversionFactor").GetData(fPeakVEMConversionFactor);
    topB.GetChild("chargeFitShoulderHeadRatio").GetData(fChargeFitShoulderHeadRatio);
    topB.GetChild("chargeFitChi2Accept").GetData(fChargeFitChi2Accept);
    topB.GetChild("chargeVEMConversionFactor").GetData(fChargeVEMConversionFactor);
    topB.GetChild("onlineChargeVEMFactor").GetData(fOnlineChargeVEMFactor);
    topB.GetChild("chargeMIPConversionFactor").GetData(fChargeMIPConversionFactor);

    Branch shapeFitB = topB.GetChild("shapeFitRange");
    shapeFitB.GetChild("beforeCalibrationVersion12").GetData(fShapeFitRangeBefore12);
    shapeFitB.GetChild("sinceCalibrationVersion12").GetData(fShapeFitRangeSince12);

    topB.GetChild("riseTimeFractions").GetData(fRiseTimeFractions);
    topB.GetChild("fallTimeFractions").GetData(fFallTimeFractions);

    const bool useUnsaturatedTraces = bool(topB.GetChild("useUnsaturatedTraces"));

    fForceLSTriggerRecalculation = bool(topB.GetChild("forceLSTriggerRecalculation"));

    fKeepLSTOTTrigger = bool(topB.GetChild("keepLSTOTTrigger"));

    {
      const auto ts =
        topB.GetChild("recalculateLSTriggerForTestStations").Get<vector<int>>();
      fRecalculateLSTriggerForTestStations = set<int>(ts.begin(), ts.end());
    }

    Branch fsB = topB.GetChild("findSignal");
    fsB.GetChild("threshold").GetData(fFindSignalThreshold);
    fsB.GetChild("binsAboveThreshold").GetData(fFindSignalBinsAboveThreshold);

    fTreatHGLGEqualInSignalSearch = bool(topB.GetChild("treatHGLGEqualInSignalSearch"));

    fApplyBackwardFlatPieceCheck = bool(topB.GetChild("applyBackwardFlatPieceCheck"));

    fDecreaseLGFlatPieceTolerance = bool(topB.GetChild("decreaseLGFlatPieceTolerance"));

    fIncludeWaterCherenkovDetectorInScintillatorStartStopDetermination =
      topB.GetChild("includeWaterCherenkovDetectorInScintillatorStartStopDetermination").Get<bool>();

    topB.GetChild("binsBeforeStartForSPMT").GetData(fBinsBeforeStartForSPMT);
    
    // calculated quantities
    fFADCSignalComponent = useUnsaturatedTraces ? StationConstants::eTotalNoSaturation : StationConstants::eTotal;

    // info stuff
    if (topB.GetChild("infoParameters")) {

      ostringstream info;
      info << "Version: "
           << GetVersionInfo(VModule::eRevisionNumber) << "\n"
              "Parameters:\n"
              "         PMTSummationCutoff: " << fPMTSummationCutoff << "\n"
              " validDynodeAnodeRatioRange: " << fValidDynodeAnodeRatioRange << "\n"
              "               peakFitRange: " << fPeakFitRange << "\n"
              "          peakFitChi2Accept: " << fPeakFitChi2Accept << "\n"
              "    peakVEMConversionFactor: " << fPeakVEMConversionFactor << "\n"
              " chargeFitShoulderHeadRatio: " << fChargeFitShoulderHeadRatio << "\n"
              "        chargeFitChi2Accept: " << fChargeFitChi2Accept << "\n"
              "  chargeVEMConversionFactor: " << fChargeVEMConversionFactor << "\n"
              "      onlineChargeVEMFactor: " << fOnlineChargeVEMFactor << "\n"
              "  chargeMIPConversionFactor: " << fChargeMIPConversionFactor << "\n"
              "          riseTimeFractions: " << fRiseTimeFractions << "\n"
              "          fallTimeFractions: " << fFallTimeFractions << "\n"
              "       useUnsaturatedTraces: " << (useUnsaturatedTraces ? "true" : "false") << "\n"
              "forceLSTriggerRecalculation: " << (fForceLSTriggerRecalculation ? "true" : "false") << "\n"
              "           keepLSTOTTrigger: " << (fKeepLSTOTTrigger ? "true" : "false") << "\n"
              "recalculateLSTriggerForTestStations:\n ";
      for (const auto& id : fRecalculateLSTriggerForTestStations)
        info << ' ' << id;
      info << "\n"
              "                  findSignal:\n"
              "                    threshold: " << fFindSignalThreshold << "\n"
              "           binsAboveThreshold: " << fFindSignalBinsAboveThreshold << "\n"
              "treatHGLGEqualInSignalSearch: " << fTreatHGLGEqualInSignalSearch << "\n"
              " applyBackwardFlatPieceCheck: " << fApplyBackwardFlatPieceCheck << "\n"
              "decreaseLGFlatPieceTolerance: " << fDecreaseLGFlatPieceTolerance;
      INFO(info);
    }

    // additional sanity checks
    if (fValidDynodeAnodeRatioRange.first < 0 ||
        fValidDynodeAnodeRatioRange.second < 0 ||
        fValidDynodeAnodeRatioRange.first >= fValidDynodeAnodeRatioRange.second) {
      ERROR("error in <validDynodeAnodeRatioRange>");
      return eFailure;
    }
    if (fRiseTimeFractions.first  < 0 || fRiseTimeFractions.first  > 1 ||
        fRiseTimeFractions.second < 0 || fRiseTimeFractions.second > 1 ||
        fFallTimeFractions.first  < 0 || fFallTimeFractions.first  > 1 ||
        fFallTimeFractions.second < 0 || fFallTimeFractions.second > 1) {

      ERROR("Rise/fall time fractions must be within [0, 1]");
      return eFailure;
    }
    if (fRiseTimeFractions.first >= fRiseTimeFractions.second ||
        fFallTimeFractions.first >= fFallTimeFractions.second ||
        fRiseTimeFractions.first >= fFallTimeFractions.second) {

      ERROR("Rise/fall time definition is not in the ascending order");
      return eFailure;
    }

    return eSuccess;
  }


  VModule::ResultFlag
  SdCalibrator::Run(evt::Event& event)
  {
    INFO(".");
    if (!event.HasSEvent())
      return eSuccess;
    SEvent& sEvent = event.GetSEvent();

    fToldYaPeak = false;
    fToldYaCharge = false;
    fToldYaShape = false;

    vector<int> noVEMStations;
    CalibrationVersionMap calibrationVersions;
    TriggerMigrationMatrix triggerMigration(0);
    vector<int> randomTrigger;
    vector<int> badCompression;
    int noTrigger = 0;

    if (sEvent.StationsBegin() != sEvent.StationsEnd())
      ++RunController::GetInstance().GetRunData().GetNamedCounters()["SdCalibrator/CalibratedEvents"];

    const bool isCommsCrisis = kCommsCrisis.IsInRange(det::Detector::GetInstance().GetTime());

    // loop on stations
    int nErrorZero = 0;
    for (auto& station : sEvent.StationsRange()) {

      if (!station.HasTriggerData()) {
        station.SetRejected(StationConstants::eNoTrigger);
        ++noTrigger;
        continue;
      }

      // T2Life() has to be checked before SetSilent() is applied!
      if (!station.IsT2Life())
        station.SetRejected(StationConstants::eNotAliveT2);
      if (isCommsCrisis && !station.IsT2Life120())
        station.SetRejected(StationConstants::eNotAliveT120);

      const StationTriggerData& trig = station.GetTriggerData();
      if (trig.GetErrorCode() & 0xff) { // for UUBs errorcodes are above 256.
        if (trig.IsSilent() && !station.IsRejected())
          station.SetSilent();
        else
          station.SetRejected(StationConstants::eErrorCode);
        continue;
      }

      if (trig.IsRandom()) {
        randomTrigger.push_back(station.GetId());
        station.SetRejected(StationConstants::eRandom);
        continue;
      }

      if (!station.HasCalibData()) {
        station.SetRejected(StationConstants::eNoCalibData);
        continue;
      }

      // exclude FADCs with Patrick's data
      if (station.GetCalibData().GetVersion() > 32000) {
        ostringstream warn;
        warn << "Station " << station.GetId() << " has LS calibration version "
             << station.GetCalibData().GetVersion() << '!';
        WARNING(warn);
        station.SetRejected(StationConstants::eNoCalibData);
        continue;
      }

      if (!station.HasGPSData()) {
        station.SetRejected(StationConstants::eNoGPSData);
        continue;
      }

      ApplyTimeCorrection(station);
      // check for "bad compressed data"
      if (sEvent.HasTrigger()) {
        const int trigSecond = sEvent.GetTrigger().GetTime().GetGPSSecond();
        const int timeDiff = int(station.GetGPSData().GetSecond()) - trigSecond;
        if (abs(timeDiff) > 1) {
          station.GetTriggerData().SetErrorCode(StationTriggerData::eBadCompress);
          station.SetRejected(StationConstants::eBadCompress);
          const int sId = station.GetId();
          badCompression.push_back(sId);
          ostringstream info;
          info << "Bad compress data: station " << sId << " has " << timeDiff
               << " sec of time difference to event trigger.";
          INFO(info);
          continue;
        }
      }

      const auto& dStation = det::Detector::GetInstance().GetSDetector().GetStation(station);
      fIsUub = dStation.IsUUB();

      CalculatePeakAndCharge(station);

      if(fIsUub && station.HasSmallPMT()){
        if(CopySmallPMTCalibData(station)){
          station.SetIsSmallPMTOk();
        } else{
          ostringstream warn;
          warn << "Station " << station.GetId() << ": No valid SmallPMT calibration.";
          station.SetIsSmallPMTOk(false);
          try {
            station.GetSmallPMTPMT().GetCalibData().SetIsTubeOk(false);
            warn << "\n Setting IsTubeOk as false for SmallPMT."; 
          } catch (NonExistentComponentException&) {/* */}
          WARNING(warn);
        }
      }
      
      if (!ComputeBaselines(station) ||
          !BuildSignals(station) ||
          !MergeSignals(station) ||
          !SelectSignal(station)) {
        station.SetRejected(StationConstants::eBadCalib);
        continue;
      }

      if (station.GetCalibData().GetVersion() <= 12 ||
          fForceLSTriggerRecalculation ||
          fRecalculateLSTriggerForTestStations.find(station.GetId()) != fRecalculateLSTriggerForTestStations.end())
        ResetStationTrigger(station, calibrationVersions, triggerMigration);

      // The default for stations is to be candidate stations
      // A station should never be reset to Candidate ever in the Module Sequence
      // because it overrules any previously done rejection not known to the SdCalibrator
      // It should only be rejected or set silent as done above or by other Modules before or after this module

      ++nErrorZero;

    } // end loop on stations

    sEvent.SetNErrorZeroStations(nErrorZero);

    String::StationIdListWithMessage(noVEMStations, "without VEM trace rejected.");
    String::StationIdListWithMessage(randomTrigger, "with random trigger rejected.");
    String::StationIdListWithMessage(badCompression, "with bad compression data rejected.");

    if (noTrigger) {
      ostringstream info;
      info << noTrigger << " station" << String::Plural(noTrigger)
           << String::OfIds(badCompression) << " without trigger data rejected.";
      INFO(info);
    }

    if (!calibrationVersions.empty()) {
      ostringstream msg;
      msg << "Trigger recalculation for LS calibration versions:";
      CalibrationVersionMap::const_iterator it = calibrationVersions.begin();
      msg << " v" << it->first << " (" << it->second << ')';
      for (++it; it != calibrationVersions.end(); ++it)
        msg << ", v" << it->first << " (" << it->second << ')';
      msg << "\nTrigger migrations (old : new trigger):\n";
      static const char* trigs[] = { "Non", "Th1", "MoPS","TOTd", "Th2", "TOT" };
      TabularStream tab("r|r|r|r|r|r|r");
      tab << "o\\n";
      for (int i = 0; i < 6; ++i)
        tab << endc << trigs[i];
      tab << endr << hline;
      for (int i = 0; i < 6; ++i) {
        tab << trigs[i];
        for (int j = 0; j < 6; ++j) {
          tab << endc;
          if (triggerMigration[i][j])
            tab << triggerMigration[i][j];
        }
        tab << endr;
      }
      tab << delr;
      msg << tab;
      INFO(msg);
    }

    return eSuccess;
  }


  VModule::ResultFlag
  SdCalibrator::Finish()
  {
    return eSuccess;
  }


  void
  SdCalibrator::ResetStationTrigger(Station& station,
                                    CalibrationVersionMap& calibrationVersions,
                                    TriggerMigrationMatrix& triggerMigration)
    const
  {
    vector<const TraceI*> validFadcTraces;
    vector<double> validVems;
    vector<double> validBaselines;
    for (const auto& pmt : station.PMTsRange(sdet::PMTConstants::eAnyType)) {
      if (pmt.HasFADCTrace() && pmt.GetCalibData().IsTubeOk() && pmt.HasRecData()) {
        validFadcTraces.push_back(&pmt.GetFADCTrace(sdet::PMTConstants::eHighGain));
        validVems.push_back(pmt.GetRecData().GetVEMPeak());
        validBaselines.push_back(pmt.GetCalibData().GetBaseline());
      }
    }
    const unsigned int n = validFadcTraces.size();

    // we need at least one trace
    if (n < 1)
      return;

    const auto& dStation = det::Detector::GetInstance().GetSDetector().GetStation(station);

    // threshold levels depend on the number of available PMTs
    // see the Trigger and Aperture... paper in NIM
    // these thresholds are in online VEM (raw VEM) aka omnidirectional peak
    double t1Threshold = 0;
    double t2Threshold = 0;
    switch (n) {
    case 1:
      t1Threshold = 2.8;
      t2Threshold = 4.5;
      break;
    case 2:
      t1Threshold = 2;
      t2Threshold = 3.8;
      break;
    default:
      t1Threshold = 1.75;
      t2Threshold = 3.2;
      break;
    }
    const int uubScaling = station.IsUUB() ? 4 : 1;
    vector<int> pmtT1Thresholds;
    vector<int> pmtT2Thresholds;
    vector<int> pmtToTThresholds;
    for (unsigned int i = 0; i < n; ++i) {
      // we have to convert the thresholds from VEM to FADC
      const auto base = validBaselines[i];
      const auto vem = uubScaling * validVems[i] / fPeakVEMConversionFactor;
      pmtT1Thresholds.push_back(round(base + vem * t1Threshold));
      pmtT2Thresholds.push_back(round(base + vem * t2Threshold));
      pmtToTThresholds.push_back(round(base + vem * 0.2/*VEM*/));
    }

    const unsigned int twoOrLess = (n > 2) ? 2 : n;
    using namespace sdet::Trigger;
    const ThresholdUp t1th(pmtT1Thresholds, n, n);
    const ThresholdUp t2th(pmtT2Thresholds, n, n);
    const TimeOverThreshold tot(pmtToTThresholds, 12, 120, twoOrLess, n);
    const int adcFactor = station.IsUUB() ? 4 : 1;
    const double integralThreshold = adcFactor * 75;
    const DecayingIntegral integ(122, 0.5, integralThreshold, validBaselines, 3./4, 1./64, n);
    const TimeOverThresholdDeconvolved totd(pmtToTThresholds, 45, 54,  // see GAP-2018-001
                                            10, 120, twoOrLess, n);
    const MultiplicityOfPositiveSteps mops(4, 121, 3, 31, 3, twoOrLess, n);

    const unsigned int traceSize = dStation.GetFADCTraceLength();
    const int latchBin = station.IsUUB() ? 660 : 246;
    sdet::StationTriggerAlgorithm stationTrigger(t1th, t2th, tot, integ, totd, mops, latchBin, traceSize);
    const auto infos = stationTrigger.Run(0, traceSize, validFadcTraces);

    if (infos.empty()) {
      WARNING("Trigger recalculation gave no result. This should not happen!");
      return;
    }
    ++calibrationVersions[station.GetCalibData().GetVersion()];

    StationTriggerData& trigger = station.GetTriggerData();

   const int oldTrigger =
     trigger.IsTimeOverThreshold() ? 5 :
     trigger.IsT2Threshold() ? 4 :
     trigger.IsMultiplicityOfPositiveSteps() ? 2 :
     trigger.IsTimeOverThresholdDeconvoluted() ? 3 :
     trigger.IsT1Threshold() ? 1 : 0;

    const sdet::StationTriggerInfo& info = infos.front();

    int newTrigger =
      info.IsTimeOverThreshold() ? 5 :
      info.IsT2Threshold() ? 4 :
      info.IsMultiplicityOfPositiveSteps() ? 2 :
      info.IsTimeOverThresholdDeconvoluted() ? 3 :
      info.IsT1Threshold() ? 1 : 0;

    const bool isOldTriggerTOT = (trigger.IsTimeOverThreshold() ||
                                  trigger.IsTimeOverThresholdDeconvoluted() ||
                                  trigger.IsMultiplicityOfPositiveSteps());

    if (fKeepLSTOTTrigger && isOldTriggerTOT)
      newTrigger = oldTrigger;

    ++triggerMigration[oldTrigger][newTrigger];

    //trigger.SetPLDTrigger(info.GetPLDBits()); change the bits also? (HD)
    trigger.SetAlgorithm(info.GetTrigger());
  }


  void
  SdCalibrator::ApplyTimeCorrection(Station& station)
  {
    StationGPSData& gpsData = station.GetGPSData();

    // NEW : TAP 26/04/2003 -> From CDAS v1r2 : taking into account Offsets...
    // Warning, part of the field is used for the tick offset:
    // GPS Offset = 0.01*(short)(gps->Offset & 0xffff)
    // Tick Offset = (short)(gps->Offset>>16)
    // New: taking into account 100ns jumps
    // From Moulin Rouge and Dia Noche we found that the TickFall-Tick
    // can be 0, 9, 10, 11 or a big number. The big number could be
    // understood if it is the trigger of another event. It was found
    // that if the dt is 0, there is a 100ns jump in the event, and not
    // in any other case, including big values. Hence this empiric
    // correction
    //
    // This is the code from IoSd v1r2 :
    // gps->NanoSecond =
    // (unsigned int)((gps->Tick*(1000000000.0 + gps->NextST - gps->CurrentST)
    // /gps->Next100) + gps->CurrentST + 0.01*(short)(gps->Offset & 0xffff))
    // -100*(gps->TickFall == gps->Tick);

    const unsigned int tick = gpsData.GetTick();
    const int currentST = gpsData.GetCurrentST();
    const int next100 = gpsData.GetNext100();
    const int nextST = gpsData.GetNextST();
    
#ifndef IOSD_V1R0
    const unsigned int tickFall = gpsData.GetTickFall();
    const int offset = gpsData.GetOffset();

    const unsigned int nanosecond =
      (unsigned int)((tick * (1e9 + nextST - currentST) / next100) + currentST +
                     0.01 * short(offset & 0xffff)) - 100 * (tickFall == tick);
#else
    const unsigned int nanosecond =
      (unsigned int)((tick * (1e9 + nextST - currentST) / next100) + currentST);
#endif
    
    gpsData.SetCorrectedNanosecond(nanosecond);
  }


  bool
  SdCalibrator::CopySmallPMTCalibData(Station& station)
  {
    // Check specific SmallPMT quantities
    const auto& smallpmt = station.GetSmallPMT();
    if(!smallpmt.HasCalibData())
      return false;
    
    const auto& spmtCalib = smallpmt.GetCalibData();
    
    if(!spmtCalib.IsTubeOk())
      return false;
    
    try {
      
      // Check standard PMT quantities
      auto& pmt = station.GetSmallPMTPMT();
      if (!pmt.HasCalibData())
        return false;

      const auto& pmtCalibData = pmt.GetCalibData();

      if (!pmtCalibData.IsTubeOk())
        return false;
      
      if (!pmt.HasRecData())
        pmt.MakeRecData();
      auto& pmtRec = pmt.GetRecData();
    
      double spmtVemCharge = 1./(spmtCalib.GetBeta()*spmtCalib.GetCorrectionFactor());
      double spmtVemChargeErr = (spmtCalib.GetBetaError()*spmtCalib.GetCorrectionFactor()) *
        spmtVemCharge * spmtVemCharge;
      
      if(!std::isnan(spmtVemCharge) && !std::isinf(spmtVemCharge) &&
         spmtVemCharge>0 && spmtVemChargeErr>0){
        
        pmtRec.SetVEMCharge(spmtVemCharge, spmtVemChargeErr);

        // SmallPMT "VEMpeak" estimation
        double muonAreaOverPeak = 0.;
        for (const auto& lpmt : station.PMTsRange()) {            
          if (lpmt.HasCalibData() &&
              lpmt.GetCalibData().IsTubeOk() &&
              lpmt.HasRecData() &&
              lpmt.GetRecData().GetVEMPeak() > 0)
            muonAreaOverPeak +=
              lpmt.GetRecData().GetVEMCharge()/lpmt.GetRecData().GetVEMPeak();
        }
        if(muonAreaOverPeak>0)                      
          pmtRec.SetVEMPeak(spmtVemCharge/muonAreaOverPeak);
        else{
          ostringstream warn;
          warn << "Station " << station.GetId()
               << ": Cannot calculate properly SmallPMT VEMpeak. Using default value = 1/3";
          WARNING(warn);
          pmtRec.SetVEMPeak(1./3.);
        }
        
        ostringstream msg;
        msg << "SmallPMT calibration for station " << station.GetId() << "\n"
            << " beta = " << spmtCalib.GetBeta() << " +- " << spmtCalib.GetBetaError()
            << " ---> VEMcharge = " << pmtRec.GetVEMCharge() << " +- " << pmtRec.GetVEMChargeError()
            << " (VEMpeak = " << pmtRec.GetVEMPeak() << ")";
        //INFO(msg);

        pmtRec.SetVEMChargeFromHistogram(false);
        pmtRec.SetVEMPeakFromHistogram(false);
        
        pmtRec.SetMuonChargeSlope(0);
        pmtRec.SetMuonPulseDecayTime(0,0);
        pmtRec.SetDynodeAnodeRatio(1);  
        
      } else {
        return false;
      }
      
    } catch (NonExistentComponentException& ex) {
      WARNING(ex.GetMessage());
      return false;
    }
      
    return true;
  }
  

  void
  SdCalibrator::CalculatePeakAndCharge(Station& station)
  {
    const auto& stationCalibData = station.GetCalibData();

    const auto calibVersion = stationCalibData.GetVersion();
    const auto& dStation = det::Detector::GetInstance().GetSDetector().GetStation(station);

    typedef VariableBinHistogramWrap<short, int> CalibHistogram;
    const auto& peakHistoBinning = dStation.GetMuonPeakHistogramBinning<short>();
    const auto& chargeHistoBinning = dStation.GetMuonChargeHistogramBinning<short>();

    for (auto& pmt : station.PMTsRange(sdet::PMTConstants::eAnyType)) {

      // No muon histograms for SmallPMT.
      // It has its own dedicated function "CopySmallPMTCalibData"
      // to fill VEM values in RecData from SmallPMTCalibData class.
      if (pmt.GetType() == sdet::PMTConstants::eWaterCherenkovSmall)
        continue;
      
      if (!pmt.HasCalibData())
         continue;

      const auto& pmtCalibData = pmt.GetCalibData();

      if (!pmtCalibData.IsTubeOk())
        continue;

      if (!pmt.HasRecData())
        pmt.MakeRecData();
      auto& pmtRec = pmt.GetRecData();

#ifdef DO_PEAK_HISTOS
      if (!pmtRec.GetVEMPeak()) {
        // load online calibration value
        const double rawPeak = pmtCalibData.GetVEMPeak();
        // null approximation
        double vemPeak = rawPeak / fPeakVEMConversionFactor;
        pmtRec.SetOnlineVEMPeak(vemPeak);
        bool vemPeakFromHisto = false;

        // try to improve values above with histogram fitting
        if (calibVersion > 8) {
          // DV: peak seems not to be of the interest any more, simple estimate only
          // analyze peak histogram
          // DV: peak seems not to be of the interest to cdas people any more,
          // they prefer simple LS estimate only, maybe we should not follow
          // since this is the value that vemizes the trace and consequently
          // (together with vem charge) enters the integrated signal
          const int muonPeakHistoSize = pmtCalibData.GetMuonPeakHisto().size();
          if (!muonPeakHistoSize || int(peakHistoBinning.size())-1 != muonPeakHistoSize) {
            if (!fToldYaPeak) {
              WARNING("According to the LS calibration version there should be a muon peak histogram... Will not tell you again!");
              fToldYaPeak = true;
            }
          } else {
            const CalibHistogram peakHisto(peakHistoBinning, pmtCalibData.GetMuonPeakHisto());
            if (rawPeak > 0 && peakHisto.GetMaximum() > 200) {
              const double baseEstimate =
                pmtCalibData.GetBaseline() - pmtCalibData.GetMuonPeakHistoOffset();
              const double base = (fabs(baseEstimate) < 20) ? baseEstimate : 0;
              const double peakLow  = fPeakFitRange.first  * rawPeak;
              const double peakHigh = fPeakFitRange.second * rawPeak;

              if (peakHigh - peakLow >= 5) {
                try {
                  // note: x axis is offset by base, in order to be the same
                  // for all histograms
                  QuadraticFitData qf;
                  MakeQuadraticFitter(peakHisto,
                                      peakLow + base, peakHigh + base).GetFitData(qf);
                  pmtRec.GetMuonPeakFitData() = qf;
                  const double fittedPeak = qf.GetExtremePosition() - base;
                  // reasonable limits for the result
                  if (peakLow <= fittedPeak && fittedPeak <= peakHigh &&
                      qf.GetChi2() <= fPeakFitChi2Accept*qf.GetNdof()) {
                    vemPeak = fittedPeak / fPeakVEMConversionFactor;
                    vemPeakFromHisto = true;
                  }
                } catch (OutOfBoundException& ex) {
                  WARNING(ex.GetMessage());
                }
              }
            }
          }
        }
        pmtRec.SetVEMPeak(vemPeak);
        pmtRec.SetVEMPeakFromHistogram(vemPeakFromHisto);
      }
#endif

      if (!pmtRec.GetVEMCharge()) {
        // load online calibration value
        const double rawCharge = pmtCalibData.GetVEMCharge() * fOnlineChargeVEMFactor;
        //                                                 DV: ^^^^^ is from hump to LS
        // null approximation
        // The charge conversion factor is the correction for the difference between the
        // positions of the peak of charge histograms for the VEM (vertical,central muons) or
        // MIP (vertical, muons) and the peak of the charge histogram for omni-directional background
        // particles.
        // charge conversion factor depends on PMT type, so far eScintillator, and eWaterCherenkovLarge;
        const double chargeConversionFactor =
          (pmt.GetType() == sdet::PMTConstants::eScintillator) ? fChargeMIPConversionFactor : fChargeVEMConversionFactor;
        double vemCharge = rawCharge / chargeConversionFactor;
        pmtRec.SetOnlineVEMCharge(vemCharge);
        double vemChargeErr = 20 / chargeConversionFactor;
        bool vemChargeFromHisto = false;
        double muChargeSlope = 0;

        // try to improve values above with histogram fitting
        if (calibVersion > 8) {
          const int muonChargeHistoSize = pmtCalibData.GetMuonChargeHisto().size();
          // analyze charge histogram
          if (!muonChargeHistoSize || int(chargeHistoBinning.size()) - 1 != muonChargeHistoSize) {
            if (!fToldYaCharge) {
              WARNING("According to the LS calibration version there should be a muon charge histogram... Will not tell you again!");
              fToldYaCharge = true;
            }
          } else {
            const CalibHistogram chargeHisto(chargeHistoBinning, pmtCalibData.GetMuonChargeHisto());
            // apparently there were 19 bins in calibration version 13
            const double baseEstimate = pmtCalibData.GetBaseline() *
              (calibVersion == 13 ? 19 : 20) - pmtCalibData.GetMuonChargeHistoOffset();
            const double base = (fabs(baseEstimate) < 20) ? baseEstimate : 0;
            
            if (chargeHisto.GetMaximum() > 500) {

              // skip 5 last bins and any small values at the high end
              const int size = chargeHisto.GetNBins();
              int start = (size - 1) - 5;

              {
                const auto small = CalibrationParameters::GetChargeHistogramSmallThreshold(fIsUub);
                for ( ; start >= 2 && chargeHisto.GetBinAverage(start) < small; --start)
                  ;
              }

              // find "head-and-shoulder": from the upper side of the histogram,
              // search for local maximum (head), surrounded by drops (shoulders)
              // with shoulder/head value ratio of less than
              // fChargeWindowShoulderHeadRatio
              int maxBin = start;
              double maxValue = chargeHisto.GetBinAverage(start);
              int shoulderLow = 0;
              {
                double value = chargeHisto.GetBinAverage(start - 1);
                double value1 = chargeHisto.GetBinAverage(start - 2);
                for (int pos = start - 1; pos >= 2; --pos) {
                  if (maxValue < value) {
                    maxValue = value;
                    maxBin = pos;
                  }
                  const double reducedMax = maxValue * fChargeFitShoulderHeadRatio;
                  // require 3 consecutive values to be lower than reducedMax
                  // to qualify as a shoulder
                  const double value2 = chargeHisto.GetBinAverage(pos - 2);
                  if (value <= reducedMax && value1 <= reducedMax && value2 <= reducedMax) {
                    shoulderLow = pos;
                    break;
                  }
                  value = value1;
                  value1 = value2;
                }
              }
              if (shoulderLow) {
                // find upper shoulder
                const double reducedMax = maxValue * fChargeFitShoulderHeadRatio;
                const int size2 = size - 2;

                int shoulderHigh = 0;
                {
                  double value = chargeHisto.GetBinAverage(maxBin + 1);
                  double value1 = chargeHisto.GetBinAverage(maxBin + 2);
                  for (int pos = maxBin + 1; pos < size2; ++pos) {
                    const double value2 = chargeHisto.GetBinAverage(pos+2);
                    if (value <= reducedMax && value1 <= reducedMax && value2 <= reducedMax) {
                      shoulderHigh = pos;
                      break;
                    }
                    value = value1;
                    value1 = value2;
                  }
                }

                if (shoulderHigh) {
                  const double chargeLow  = chargeHisto.GetBinCenter(shoulderLow);
                  const double chargeHigh = chargeHisto.GetBinCenter(shoulderHigh);

                  if (chargeLow <= rawCharge && rawCharge <= chargeHigh) {
                    try {
                      // now fit in shoulder window
                      QuadraticFitData qf;
                      MakeQuadraticFitter(chargeHisto, chargeLow, chargeHigh).GetFitData(qf);
                      pmtRec.GetMuonChargeFitData() = qf;
                      const double fittedCharge = qf.GetExtremePosition() - base;

                      // reasonable limits for the result
                      if (chargeLow <= fittedCharge && fittedCharge <= chargeHigh &&
                          qf.GetChi2() <= fChargeFitChi2Accept*qf.GetNdof()) {
                        vemCharge = fittedCharge / chargeConversionFactor;
                        vemChargeErr = qf.GetExtremePositionError() / chargeConversionFactor;
                        vemChargeFromHisto = true;
                      }
                    } catch (OutOfBoundException& ex) {
                      ostringstream warn;
                      warn << ex.GetMessage() << "\n"
                              "Quadratic fit between bins " << chargeLow << " and " << chargeHigh
                           << " failed on this charge histogram:\n";
                      for (unsigned int i = 0, n = chargeHisto.GetNBins(); i < n; ++i)
                        warn << chargeHisto.GetBinCenter(i) << ' ' << chargeHisto.GetBinAverage(i) << '\n';
                      WARNING(warn);
                    }

                    try {
                      // slope of muon charge
                      ExponentialFitData ef;
                      const double start = chargeHigh + 10;
                      const double stop = min(2*start, 950.);
                      if (start+10 < stop) {
                        MakeExponentialFitter(chargeHisto, start, stop).GetFit(ef);
                        pmtRec.GetMuonChargeSlopeFitData() = ef;

                        const double slope = vemCharge * ef.GetSlope();

                        if (slope < -0.5)
                          muChargeSlope = slope;
                      }
                    } catch (OutOfBoundException& ex) {
                      WARNING(ex.GetMessage());
                    }
                  }
                }
              }
            } // if max > 500
          }
        }
        pmtRec.SetVEMCharge(vemCharge, vemChargeErr);
        pmtRec.SetVEMChargeFromHistogram(vemChargeFromHisto);
        pmtRec.SetMuonChargeSlope(muChargeSlope);
      }

      // shape histogram
      double muDecayTime = 0;
      double muDecayTimeErr = 0;
      if (calibVersion > 8) {
        // muon decay time from muon shape histogram
        if (pmtCalibData.GetMuonShapeHisto().empty()) {
          if (!fToldYaShape) {
            WARNING("According to the LS calibration version there should be a muon shape histogram... Will not tell you again!");
            fToldYaShape = true;
          }
        } else {
          const vector<int>& muShapeHisto = pmtCalibData.GetMuonShapeHisto();
          const unsigned int size = muShapeHisto.size();
          if (size) {
            try {
              ExponentialFitData ef;

              typedef VariableBinHistogramWrap<double, int> ShapeHistogram;
              const vector<double> shapeHistoBinning =
                dStation.GetMuonShapeHistogramBinning(calibVersion);
              const double subtract = muShapeHisto[0];
              MakeExponentialFitter(ShapeHistogram(shapeHistoBinning, muShapeHisto),
                                    (calibVersion < 12 ?
                                      fShapeFitRangeBefore12 :
                                      fShapeFitRangeSince12)).GetFit(ef, subtract);
              pmtRec.GetMuonShapeFitData() = ef;

              const double slope = ef.GetSlope();
              const double slopeErr = ef.GetSlopeError();

              if (slope) {
                const double decayTime = -1 / slope;
                if (0 <= decayTime && decayTime < 1000*nanosecond) {
                  muDecayTime = decayTime;
                  muDecayTimeErr = slopeErr / Sqr(slope);
                }
              }
            } catch (OutOfBoundException& ex) {
              WARNING(ex.GetMessage());
            }
          }
        }
      }
      pmtRec.SetMuonPulseDecayTime(muDecayTime, muDecayTimeErr);

      const double dynodeAnodeRatio =
        pmtCalibData.GetDynodeAnodeRatio() / (calibVersion == 12 ? 1.07 : 1);
      pmtRec.SetDynodeAnodeRatio(dynodeAnodeRatio);

    }
  }


  // set rise/fall time for PMT trace and return shape parameter
  void
  SdCalibrator::ComputeShapeRiseFallPeak(PMTRecData& pmtRec,
                                         const double binTiming,
                                         const unsigned int startBin,
                                         const unsigned int startIntegration,
                                         const unsigned int endIntegration,
                                         const double traceIntegral)
    const
  {
    // sorry for not using the nice TraceAlgorithms, but all this values can
    // be calculated in single trace pass

    if (traceIntegral <= 0)
      return;

    const double riseStartSentry = fRiseTimeFractions.first * traceIntegral;
    const double riseEndSentry = fRiseTimeFractions.second * traceIntegral;
    const double fallStartSentry = fFallTimeFractions.first * traceIntegral;
    const double fallEndSentry = fFallTimeFractions.second * traceIntegral;
    const unsigned int shapeSentry =
      startIntegration + (unsigned int)(600*nanosecond / binTiming);
    const double t50Sentry = 0.5 * traceIntegral;
    double riseStartBin = 0;
    double riseEndBin = 0;
    double fallStartBin = 0;
    double fallEndBin = 0;
    double t50Bin = 0;
    double sumEarly = 0;

    double peakAmplitude = 0;
    double runningSum = 0;
    double oldSum = 0;

    const TraceD& trace = pmtRec.GetVEMTrace();

    for (unsigned int i = startIntegration; i < endIntegration; ++i) {

      const double binValue = trace[i];
      runningSum += binValue;

      if (peakAmplitude < binValue)
        peakAmplitude = binValue;

      if (!sumEarly && i >= shapeSentry)
        sumEarly = oldSum;

      if (!riseStartBin && runningSum > riseStartSentry)
        riseStartBin = i - (runningSum - riseStartSentry) / (runningSum - oldSum);

      if (!riseEndBin && runningSum > riseEndSentry)
        riseEndBin = i - (runningSum - riseEndSentry) / (runningSum - oldSum);

      if (!fallStartBin && runningSum > fallStartSentry)
        fallStartBin = i - (runningSum - fallStartSentry) / (runningSum - oldSum);

      if (!fallEndBin && runningSum > fallEndSentry)
        fallEndBin = i - (runningSum - fallEndSentry) / (runningSum - oldSum);

      if (!t50Bin && runningSum > t50Sentry)
        t50Bin = i - (runningSum - t50Sentry) / (runningSum - oldSum);

      oldSum = runningSum;

    }

    pmtRec.SetPeakAmplitude(peakAmplitude);
    pmtRec.SetRiseTime(binTiming * (riseEndBin-riseStartBin), 0);
    pmtRec.SetFallTime(binTiming * (fallEndBin-fallStartBin), 0);
    pmtRec.SetT50(binTiming * (t50Bin-startBin));
    if (shapeSentry < endIntegration) {
      const double sumLate = runningSum - sumEarly;
      if (sumLate > 1e-3)
        pmtRec.SetShapeParameter(sumEarly / sumLate);
    }
  }


  void
  SdCalibrator::SumPMTComponents(Station& station)
    const
  {
    // Start bin has been found and total trace and timing set. Copy individual
    // component trace information TAP - 01/02/2006.

    vector<const TraceD*> compTrace;

    const sdet::Station& dStation =
      det::Detector::GetInstance().GetSDetector().GetStation(station);

    for (unsigned int comp = 1; comp <= StationConstants::eLastSource; ++comp) {

      const StationConstants::SignalComponent component =
        static_cast<StationConstants::SignalComponent>(comp);

      compTrace.clear();

      // is component present?
      for (const auto& pmt : station.PMTsRange())
        if (pmt.HasRecData()) {
          const PMTRecData& pmtRec = pmt.GetRecData();
          if (pmtRec.HasVEMTrace(component))
            compTrace.push_back(&pmtRec.GetVEMTrace(component));
        }

      const unsigned int nPMTs = compTrace.size();

      if (nPMTs) {

        const unsigned int fadcTraceLength = dStation.GetFADCTraceLength();

        TraceD sumTrace(fadcTraceLength, dStation.GetFADCBinSize());

        for (unsigned int pos = 0; pos < fadcTraceLength; ++pos) {

          double& sum = sumTrace[pos];

          sum = 0;
          int n = 0;
          for (unsigned int pmtIndex = 0; pmtIndex < nPMTs; ++pmtIndex) {
            const double value = (*compTrace[pmtIndex])[pos];
            if (value > fPMTSummationCutoff) {
              sum += value;
              ++n;
            }
          }
          if (n)
            sum /= n;

        }

        if (!station.HasVEMTrace(component))
          station.MakeVEMTrace(component);
        station.GetVEMTrace(component) = sumTrace;

      }

    }
  }


  void
  SdCalibrator::MakeFlatBaseline(PMT& pmt, const sdet::PMTConstants::PMTGain gain)
    const
  {
    PMTRecData& pmtRec = pmt.GetRecData();
    if (!pmtRec.HasFADCBaseline(gain))
      pmtRec.MakeFADCBaseline(gain);
    TraceD& baseline = pmtRec.GetFADCBaseline(gain);
    const double onlineBaseline = pmt.GetCalibData().GetBaseline(gain);
    const int n = baseline.GetSize();
    for (int i = 0; i < n; ++i)
      baseline[i] = onlineBaseline;
  }


  bool
  SdCalibrator::ComputeBaselines(Station& station)
    const
  {
    int doneSome = 0;
    for (auto& pmt : station.PMTsRange(sdet::PMTConstants::eAnyType))
      doneSome += ComputeBaseline(station, pmt, sdet::PMTConstants::eHighGain) +
                  ComputeBaseline(station, pmt, sdet::PMTConstants::eLowGain);

    return doneSome;
  }



  bool
  SdCalibrator::ComputeBaseline(const Station& station, PMT& pmt, const sdet::PMTConstants::PMTGain gain)
    const
  {
    const sdet::Station& dStation =
      det::Detector::GetInstance().GetSDetector().GetStation(station);

    if (!pmt.HasFADCTrace() || !pmt.GetCalibData().IsTubeOk())
      return false;

    if (gain == sdet::PMTConstants::eLowGain && !pmt.GetCalibData().IsLowGainOk())
      return false;

    const TraceI& trace = pmt.GetFADCTrace(gain, fFADCSignalComponent);
    const int traceLength = trace.GetSize();

    if (!pmt.HasRecData())
      pmt.MakeRecData();
    PMTRecData& pmtRec = pmt.GetRecData();

    PMTRecData::FlatPieceCollection& flatPieces = pmtRec.GetBaselineFlatPieces(gain);
    flatPieces.clear();

    int minLength = CalibrationParameters::GetMinLength(fIsUub);

    const int saturationValue = (fFADCSignalComponent == StationConstants::eTotalNoSaturation) ?
      numeric_limits<int>::max() : dStation.GetSaturationValue();

    // increase signal variability
    bool seenSaturation = false;
    bool hitsZero = false;
    int sigma = (fDecreaseLGFlatPieceTolerance && gain == sdet::PMTConstants::eLowGain) ? 1 : 2;

    do {
      ++sigma;

      int startBin = 0;
      int stopBin = 0;
      do {

        const int startValue = trace[startBin];

        // find sigma-flat piece
        while (stopBin < traceLength) {
          if (abs(startValue - trace[stopBin]) > sigma)
            break;
          ++stopBin;
        }

        if (startValue >= saturationValue) {
          if (!startBin && stopBin == traceLength) {
            // whole trace is saturated
            flatPieces.push_back(PMTRecData::Piece(0, traceLength));
            seenSaturation = true;
            ostringstream warn;
            warn << "Station " << station.GetId() << ", PMT " << pmt.GetId() << ", "
                 << (gain == sdet::PMTConstants::eHighGain ? "high" : "low")
                 << " gain (saturated): Whole trace saturated.";
            WARNING(warn);
            break;
          } else {
            // start again
            stopBin = startBin;
            minLength = 0.25 * CalibrationParameters::GetMinLength(fIsUub);
            if (sigma < 4)
              sigma = 4;
            seenSaturation = true;
          }
        }
        if (!startValue && seenSaturation) {
          // (under)saturation of undershoot: baseline is 0 till the end
          flatPieces.push_back(PMTRecData::Piece(stopBin, traceLength));
          startBin = stopBin = traceLength;
          hitsZero = true;
        }
        if (stopBin-startBin < minLength) {
          // nothing useful found
          ++startBin;
          stopBin = startBin;
        } else {
          // RB : propagate from end of flat piece back, window centered, to check
          // if the start is found back, if not, try next bin as start
          if (fApplyBackwardFlatPieceCheck) {

            const int reverseStartValue =
              accumulate(&trace[startBin], &trace[stopBin], 0) / (stopBin - startBin);
            int reverseBin = stopBin - 1;
            while (reverseBin > startBin) {
              if (abs(reverseStartValue - trace[reverseBin]) > sigma)
                break;
              --reverseBin;
            }

            if (reverseBin == startBin) {
              flatPieces.push_back(PMTRecData::Piece(startBin, stopBin));
              startBin = stopBin;
            } else {
              ++startBin;
              stopBin = startBin;
            }

          } else {
            flatPieces.push_back(PMTRecData::Piece(startBin, stopBin));
            startBin = stopBin;
          }

        }

      } while (stopBin < traceLength);

      // should have some pieces
      if (!flatPieces.empty() &&
          flatPieces[0].first > CalibrationParameters::GetUsefulBins(fIsUub) &&
          sigma < 5) {
        // try again with larger sigma
        flatPieces.clear();
        sigma = 4;
        ostringstream warn;
        warn << "Station " << station.GetId() << ", PMT " << pmt.GetId() << ", "
             << (gain == sdet::PMTConstants::eHighGain ? "high" : "low") << " gain: "
                "No useful baseline found in the first "
             << CalibrationParameters::GetUsefulBins(fIsUub) << " bins.";
        WARNING(warn);
      }

    } while (flatPieces.empty() && sigma <= saturationValue);

    if (flatPieces.empty()) {
      MakeFlatBaseline(pmt, gain);
      if (seenSaturation)
        pmtRec.SetFADCSaturatedBins(-1, gain);
      ostringstream warn;
      warn << "Station " << station.GetId() << ", PMT " << pmt.GetId() << ", "
           << (gain == sdet::PMTConstants::eHighGain ? "high" : "low") << " gain"
           << (seenSaturation ? " (saturated):" : ":")
           << "No baseline found; using LS value.";
      WARNING(warn);
      return false;
    }

    // compute baselines
    vector<double> flatPieceMean;
    flatPieceMean.reserve(flatPieces.size());
    double meanErrorMaxPiece = 0;
    int maxPieceLength = 0;
    for (const auto& fp : flatPieces) {
      const Accumulator::SampleStandardDeviation sigma =
        for_each(&trace[fp.first], &trace[fp.second], Accumulator::SampleStandardDeviation());
      flatPieceMean.push_back(sigma.GetAverage());
      if (sigma.GetN() > maxPieceLength) {
        maxPieceLength = sigma.GetN();
        meanErrorMaxPiece = sigma.GetStandardDeviation();
      }
    }

    if (hitsZero)
      flatPieceMean.back() = 0;

    pmtRec.SetFADCBaselineError(meanErrorMaxPiece, gain);

    if (sigma > 3 && !seenSaturation) {
      ostringstream warn;
      warn << "Station " << station.GetId() << ", PMT " << pmt.GetId() << ", "
           << (gain == sdet::PMTConstants::eHighGain ? "high" : "low") << " gain: "
              "Noisy baseline, sigma = " << sigma << ", RMS = " << meanErrorMaxPiece;
      WARNING(warn);
    }

    pmtRec.SetFADCBaselineWindow(sigma, gain);

    if (!pmtRec.HasFADCBaseline(gain))
      pmtRec.MakeFADCBaseline(gain);
    TraceD& baseline = pmtRec.GetFADCBaseline(gain);

    // this comes from the Torino PMT baseline study
    // done by Simone Maldera and Gianni Navarra, GAP-2005-006 and GAP-2005-025
    const double recoveryFactor = 0.000158;

    double previousBaseline = flatPieceMean[0];

    // beginning of online baseline, before first piece
    {
      double charge = 0;
      for (int i = flatPieces[0].first - 1; i >= 0; --i) {
        const double signal = trace[i] - previousBaseline;
        if (signal > 0)
          charge += signal;
        baseline[i] = previousBaseline + charge * recoveryFactor;
      }
    }

    // first piece
    for (unsigned int i = flatPieces[0].first; i < flatPieces[0].second; ++i)
      baseline[i] = previousBaseline;

    // hole-centric: previous piece | hole | next piece
    const unsigned int nPieces = flatPieces.size();
    for (unsigned int p = 1; p < nPieces; ++p) {
      const double nextBaseline = flatPieceMean[p];
      const int start = flatPieces[p-1].second;
      const int stop = flatPieces[p].first;
      const int holeLength = stop - start;
      const double deltaBaselinePerBin =
        (nextBaseline - previousBaseline) / holeLength;
      // charge in the hole
      double charge = 0;
      for (int i = start ; i < stop; ++i) {
        const double base = previousBaseline + (i - start) * deltaBaselinePerBin;
        const double signal = trace[i] - base;
        if (signal > 0)
          charge += signal;
      }
      const double totalCharge = charge;
      if (totalCharge / holeLength < 2) {
        // linear interpolation over bins
        for (int i = start ; i < stop; ++i)
          baseline[i] = previousBaseline + (i - start) * deltaBaselinePerBin;
      } else {
        const double deltaBaselinePerCharge =
          (nextBaseline - previousBaseline) / totalCharge;
        // interpolate in the hole according to the charge increase
        charge = 0;
        for (int i = start ; i < stop; ++i) {
          const double base = previousBaseline + (i - start) * deltaBaselinePerBin;
          const double signal = trace[i] - base;
          if (signal > 0)
            charge += trace[i] - base;
          baseline[i] = previousBaseline + charge * deltaBaselinePerCharge;
        }
      }
      // fill next piece
      for (unsigned int i = stop; i < flatPieces[p].second; ++i)
        baseline[i] = nextBaseline;
      previousBaseline = nextBaseline;
    }

    // fill end (if not there already)
    {
      double charge = 0;
      for (int i = flatPieces[nPieces-1].second; i < traceLength; ++i) {
        const double signal = trace[i] - previousBaseline;
        if (signal > 0)
          charge += signal;
        baseline[i] = previousBaseline - charge * recoveryFactor;
      }
    }

    if (seenSaturation)
      pmtRec.SetFADCSaturatedBins(-1, gain);

    // DV: fix for cyclon boards that send register data in the last 8 bins
    // of FADC traces: solution is to set baseline equal to FADC values in
    // these last bins so that the signal there becomes zero.
    if ((fRecalculateLSTriggerForTestStations.find(station.GetId()) !=
         fRecalculateLSTriggerForTestStations.end()) &&
        (traceLength == int(dStation.GetFADCTraceLength()))) {
      const int registersStart = int(dStation.GetFADCTraceLength()) - 8;
      for (int i = registersStart; i < int(dStation.GetFADCTraceLength()); ++i)
        baseline[i] = trace[i];
    }

    return true;
  }

  bool
  SdCalibrator::MakeComponentVEMTraces(PMT& pmt)
    const
  {
    PMTRecData& pmtRec = pmt.GetRecData();
    const auto gainUsed = pmtRec.GetGainUsed();
    const double vemFactor =
      (gainUsed == sdet::PMTConstants::eHighGain ? 1 : pmtRec.GetDynodeAnodeRatio()) /
        pmtRec.GetVEMPeak();
    const auto& multiFADCTrace = pmt.GetMultiFADCTrace(gainUsed);
    if (multiFADCTrace.GetNLabels() < 2)
      return false;

    bool didComponents = false;
    for (const auto& component : multiFADCTrace) {
      const auto comp =
        static_cast<sevt::StationConstants::SignalComponent>(component.GetLabel());

      if (comp == sevt::StationConstants::eTotal)
        continue;

      if (!pmtRec.HasVEMTrace(comp))
        pmtRec.MakeVEMTrace(comp);
      auto& vemTrace = pmtRec.GetVEMTrace(comp);

      // The components are taken from the double-valued traces.
      const auto& fadcTrace = pmt.GetFADCTraceD(gainUsed, comp);
      const int n = fadcTrace.GetSize();
      const auto& baseline = pmtRec.GetFADCBaseline(gainUsed);

      // substract baseline from unsaturated trace
      if (comp == sevt::StationConstants::eTotalNoSaturation) {

        const auto& fadcTrace = pmt.GetFADCTrace(gainUsed, comp);
        const int n2 = fadcTrace.GetSize();

        for (int i = 0; i < n2; ++i)
          vemTrace[i] = (fadcTrace[i] - baseline[i]) * vemFactor;
      } else
        for (int i = 0; i < n; ++i)
          vemTrace[i] = fadcTrace[i] * vemFactor;

      didComponents = true;
    }

    return didComponents;
  }


  int
  SdCalibrator::BuildSignals(Station& station)
    const
  {
    vector<const PMT*> validPMTs;

    bool didComponents = false;

    const sdet::Station& dStation =
      det::Detector::GetInstance().GetSDetector().GetStation(station);

    for (auto& pmt : station.PMTsRange(sdet::PMTConstants::eAnyType)) {

      if (pmt.HasCalibData() && pmt.GetCalibData().IsTubeOk() && pmt.HasFADCTrace()) {

        if (BuildSignals(pmt, dStation.GetFADCTraceLength(), dStation.GetSaturationValue()) && pmt.HasRecData()) {

          if (pmt.GetType() == sdet::PMTConstants::eWaterCherenkovLarge)
            validPMTs.push_back(&(pmt)); // since there are multiple such PMTs, they are processed later
          else if (pmt.GetType() == sdet::PMTConstants::eScintillator) {
            Scintillator& scintillator = station.GetScintillator();
            if (!scintillator.HasMIPTrace())
              scintillator.MakeMIPTrace();
            else
              scintillator.GetMIPTrace().ResetAll();
            TraceD& mipTrace = scintillator.GetMIPTrace();
            const int traceLength = mipTrace.GetSize();
            const TraceD& pmtTrace = pmt.GetRecData().GetVEMTrace();
            for (int i = 0; i < traceLength; ++i)
              mipTrace[i] = pmtTrace[i];
          }

        }
        if (pmt.HasSimData() && MakeComponentVEMTraces(pmt))
          didComponents = true;
      }
    }

    if (!validPMTs.empty()) {
      const int nPMTs = validPMTs.size();
      if (!station.HasVEMTrace())
        station.MakeVEMTrace();
      else {
        // clear if it exists
        station.GetVEMTrace().ResetAll();
      }
      TraceD& vemTrace = station.GetVEMTrace();
      const int traceLength = vemTrace.GetSize();
      for (const auto& pmt : validPMTs) {
        const TraceD& pmtTrace = (pmt)->GetRecData().GetVEMTrace();
        for (int i = 0; i < traceLength; ++i)
          vemTrace[i] += pmtTrace[i] / nPMTs;
      }
    }

    if (didComponents)
      SumPMTComponents(station);

    return validPMTs.size();
  }


  bool
  SdCalibrator::BuildSignals(PMT& pmt, const unsigned int traceLength, const unsigned int trueSaturationValue)
    const
  {
    const bool isSPMT = (pmt.GetType()==sdet::PMTConstants::eWaterCherenkovSmall);
    const TraceI& highGainTrace = pmt.GetFADCTrace(sdet::PMTConstants::eHighGain, fFADCSignalComponent);
    const int saturationValue = (fFADCSignalComponent == StationConstants::eTotalNoSaturation) ?
      numeric_limits<int>::max() : trueSaturationValue;

    // check for saturation
    int highGainSaturatedBins = 0;
    for (unsigned int i = 0; i < traceLength; ++i)
      if (highGainTrace[i] >= saturationValue)
        ++highGainSaturatedBins;

    if (!pmt.HasRecData())
      pmt.MakeRecData();
    PMTRecData& pmtRec = pmt.GetRecData();
    pmtRec.SetFADCSaturatedBins(highGainSaturatedBins, sdet::PMTConstants::eHighGain);

    const TraceI& lowGainTrace = pmt.GetFADCTrace(sdet::PMTConstants::eLowGain, fFADCSignalComponent);
    PMTCalibData& pmtCalib = pmt.GetCalibData();
    const bool lgOK = pmtCalib.IsLowGainOk();

    if (lgOK) {
      int lowGainSaturatedBins = 0;
      for (unsigned int i = 0; i < traceLength; ++i)
        if (lowGainTrace[i] >= saturationValue)
          ++lowGainSaturatedBins;
      pmtRec.SetFADCSaturatedBins(lowGainSaturatedBins, sdet::PMTConstants::eLowGain);

      const auto maxBins = CalibrationParameters::GetSaturatedBinsMaximum(fIsUub);
      // The SmallPMT only presents the LowGain channel
      if(isSPMT){
        if (lowGainSaturatedBins > maxBins){
          pmtCalib.SetIsTubeOk(false);
          return false;
        }
      }
      else{      
        if (highGainSaturatedBins > maxBins || lowGainSaturatedBins > maxBins ||
            (lowGainSaturatedBins && !highGainSaturatedBins)) {
          // this is for the case where we have all or almost all of the low gain
          // saturated and high gain saturated at the same time => no useful trace,
          // or low gain saturated but not the high gain...
          pmtCalib.SetIsTubeOk(false);
          return false;
        }
      }
    } else {
      // if low gain is broken and high gain is saturated or if SPMT => no useful trace
      if (highGainSaturatedBins || isSPMT) {
        pmtCalib.SetIsTubeOk(false);
        return false;
      }
    }

    const sdet::PMTConstants::PMTGain gainUsed =
      ((highGainSaturatedBins && lgOK) || isSPMT) ? sdet::PMTConstants::eLowGain : sdet::PMTConstants::eHighGain;
    pmtRec.SetGainUsed(gainUsed);

    if (!pmtRec.HasVEMTrace())
      pmtRec.MakeVEMTrace();
    TraceD& vemTrace = pmtRec.GetVEMTrace();

    // find signal(s)
    SignalSegmentCollection& rawSignals = pmtRec.GetRawSignals();
    rawSignals.clear();

    const double vemChargeFactor = pmtRec.GetVEMPeak() / pmtRec.GetVEMCharge();

    bool isTubeOK = true;

    const double gainFactor =
      (gainUsed == sdet::PMTConstants::eLowGain) ? pmtCalib.GetDynodeAnodeRatio() : 1;
    const double gainPeakFactor = gainFactor / pmtRec.GetVEMPeak();

    const TraceI& trace =
      (gainUsed == sdet::PMTConstants::eLowGain) ? lowGainTrace : highGainTrace;
    const TraceD& baseline = pmtRec.GetFADCBaseline(gainUsed);
    const TraceD& highGainBaseline = pmtRec.GetFADCBaseline(sdet::PMTConstants::eHighGain);

    const auto findSignalThresholdMultiplier =
      CalibrationParameters::GetFindSignalThresholdMultiplier(fIsUub);
    const auto largeFADCThreshold =
      CalibrationParameters::GetLargeFADCThreshold(fIsUub);
    const auto minFADCValue =
      CalibrationParameters::GetMinFADCValue(fIsUub);
    int binsWithLargeSignal = 0;
    int binsWithSignal = 0;
    int binsOverThresh = 0;
    int start = 0;
    double max = 0;
    double charge = 0;
    for (int i = 0; i < int(traceLength); ++i) {
      const int fadc = trace[i];
      if (fadc > largeFADCThreshold)
        ++binsWithLargeSignal;
      const double fadcSignal = fadc - baseline[i];
      // before the minimum FADC value was 10 and thus it might reject small signals,
      // knowing that normal baseline fluctuations are at the level of 1-2 FADC bins,
      // and we have new triggers with small signals it is now 4
      if (fadcSignal > minFADCValue)
        ++binsWithSignal;
      const double signal = fadcSignal * gainPeakFactor;
      vemTrace[i] = signal;

      // allways on high gain, RB: not anymore
      const double testSignal =
        fTreatHGLGEqualInSignalSearch ? fadcSignal : highGainTrace[i] - highGainBaseline[i];
      if (testSignal > findSignalThresholdMultiplier * fFindSignalThreshold) {
        // first ?
        if (!binsOverThresh)
          start = i;
        ++binsOverThresh;
        charge += signal;
        if (signal > max)
          max = signal;
      } else {
        //require at least 2 bins
        if (binsOverThresh >= findSignalThresholdMultiplier * fFindSignalBinsAboveThreshold) {
          rawSignals.push_back(SignalSegment(start, i, binsOverThresh, charge * vemChargeFactor, max));
        }
        binsOverThresh = 0;
        max = 0;
        charge = 0;
      }
    }

    if (binsOverThresh >= findSignalThresholdMultiplier * fFindSignalBinsAboveThreshold) {
      rawSignals.push_back(SignalSegment(start, traceLength, binsOverThresh, charge * vemChargeFactor, max));
      if (binsWithLargeSignal > CalibrationParameters::GetBinsWithLargeSignalThreshold(fIsUub) ||
          binsWithSignal < CalibrationParameters::GetBinsWithSignalThreshold(fIsUub))
        isTubeOK = false;
    }

    if (!isTubeOK && !fIsUub) {
      pmtCalib.SetIsTubeOk(false);
      rawSignals.clear();
      return false;
    }

    SignalSegmentCollection::const_iterator rawIt = rawSignals.begin();
    if (rawIt != rawSignals.end()) {
      // joined signals
      SignalSegmentCollection& signals = pmtRec.GetSignals();
      signals.clear();
      // put first in
      signals.push_back(*rawIt);
      const auto signalMaxDist = CalibrationParameters::GetSignalMaxDist(fIsUub);
      for (++rawIt; rawIt != rawSignals.end(); ++rawIt) {
        SignalSegment& current = signals.back();
        const int dist = rawIt->fStart - current.fStop;
        const int maxDist = signalMaxDist + current.fBinsOverThresh;
        // join raw signals as long they match the conditions
        if (dist >= maxDist ||
            (0.3*rawIt->fCharge >= current.fCharge && rawIt->fMaxValue >= 5) ||
            !rawIt->fCharge)  // this one is probably not needed
          signals.push_back(*rawIt);
        else {
          // add bins inbetween
          const double addCharge =
            accumulate(&vemTrace[current.fStop], &vemTrace[rawIt->fStart], rawIt->fCharge);
          current.fCharge += addCharge * vemChargeFactor;
          current.fStop = rawIt->fStop;
          current.fBinsOverThresh += rawIt->fBinsOverThresh;
          if (current.fMaxValue < rawIt->fMaxValue)
            current.fMaxValue = rawIt->fMaxValue;
        }
      }
    }

    return true;
  }


  template<typename T1, typename T2>
  class Interval : public pair<T1, T2> {
  public:
    Interval(const pair<T1, T2>& p) : pair<T1, T2>(p) { }

    bool operator<(const Interval& interval) const
    { return this->second < interval.first; }

    bool operator==(const Interval& interval) const
    { return interval.second > this->first && interval.first < this->second; }

    void
    Merge(const Interval& interval)
    {
      if (interval.first < this->first)
        this->first = interval.first;
      if (interval.second > this->second)
        this->second = interval.second;
    }
  };


  bool
  SdCalibrator::MergeSignals(Station& station)
    const
  {
    vector<PMT*> validPMTs;
    typedef set<Interval<int, int>> Sections;
    Sections sections;

    const sdet::Station& dStation =
      det::Detector::GetInstance().GetSDetector().GetStation(station);

    for (auto& pmt : station.PMTsRange()) {

      if (pmt.HasCalibData() &&
          pmt.GetCalibData().IsTubeOk() &&
          pmt.HasRecData()) {

        validPMTs.push_back(&pmt);

        SignalSegmentCollection& signals = pmt.GetRecData().GetSignals();

        for (const auto& sig : signals) {

          Interval<int, int> newSection(make_pair(sig.fStart, sig.fStop));
          // try to insert, then merge until insert succeeds
          for (pair<Sections::iterator, bool> where; ; ) {
            where = sections.insert(newSection);
            if (!where.second) {
              // insert failed
              newSection.Merge(*where.first);
              sections.erase(where.first);
            } else
              break;
          }

        }

      }

    }

    const int nPMTs = validPMTs.size();
    SignalSegmentCollection& stationSignals = station.GetSignals();

    // we have ordered set of overlapping interval unions
    const int traceLength = dStation.GetFADCTraceLength();
    const auto findSignalThresholdMultiplier =
      CalibrationParameters::GetFindSignalThresholdMultiplier(fIsUub);
    Sections::iterator nextSectionIt = sections.begin();
    Sections::iterator currentSectionIt = nextSectionIt;
    while (currentSectionIt != sections.end()) {
      // add 10 bins at the end (if possible)
      int newStop = currentSectionIt->second + 10;
      if (newStop > traceLength)
        newStop = traceLength;
      ++nextSectionIt;
      if (nextSectionIt != sections.end() && newStop > nextSectionIt->first)
        newStop = nextSectionIt->first;
      const int start = currentSectionIt->first;
      // fill station signals
      {
        const TraceD& vemTrace = station.GetVEMTrace();
        int binsOverThresh = 0;
        double charge = 0;
        for (const auto& pmtp : validPMTs) {
          const PMT& pmt = *pmtp;
          const PMTRecData& pmtRec = pmt.GetRecData();
          const sdet::PMTConstants::PMTGain gainUsed = pmtRec.GetGainUsed();

          const TraceI& trace = fTreatHGLGEqualInSignalSearch ?
            pmt.GetFADCTrace(gainUsed, fFADCSignalComponent) : pmt.GetFADCTrace(sdet::PMTConstants::eHighGain, fFADCSignalComponent);
          const TraceD& baseline = fTreatHGLGEqualInSignalSearch ?
            pmtRec.GetFADCBaseline(gainUsed) : pmtRec.GetFADCBaseline(sdet::PMTConstants::eHighGain);

          double vemSum = 0;
          for (int i = start; i < newStop; ++i) {
            if (trace[i] - baseline[i] >= findSignalThresholdMultiplier * fFindSignalThreshold)
              ++binsOverThresh;
            vemSum += vemTrace[i];
          }
          const double factor = pmtRec.GetVEMPeak() / pmtRec.GetVEMCharge();
          charge += vemSum * factor;
        }
        stationSignals.push_back(SignalSegment(start, newStop, double(binsOverThresh)/nPMTs, charge/nPMTs));
      }
      ++currentSectionIt;
    }

    // Since the Scintillator only has one PMT, it's PMTRecData signals are the ScintillatorRecData
    // Signals and no merging is necessary. Direct copying occurs from the PMTRecData
    // SignalSegmentCollection to the ScintillatorRecData SignalSegmentCollection below.
    if (station.HasScintillator()) {
      const auto& pmt =  station.GetScintillatorPMT();
      if (pmt.HasRecData()) {
        const auto& pmtSignals = station.GetScintillatorPMT().GetRecData().GetSignals();
        auto& scintillatorSignals = station.GetScintillator().GetSignals();
        scintillatorSignals.clear();
        for (auto const signal : pmtSignals)
          scintillatorSignals.push_back(signal);
      }
    }

    return true;
  }


  bool
  SdCalibrator::SelectSignal(Station& station)
    const
  {
    const sdet::Station& dStation =
      det::Detector::GetInstance().GetSDetector().GetStation(station);

    const SignalSegmentCollection& signals = station.GetSignals();
    const unsigned int nSignals = signals.size();

    if (!nSignals) {
      // no signals found, check for saturation
      // StationRecData ctor sets all other values to zero
      for (auto& pmt : station.PMTsRange()) {
        if (pmt.HasRecData() &&
            pmt.HasCalibData() && pmt.GetCalibData().IsTubeOk()) {
          const PMTRecData& pmtRec = pmt.GetRecData();

          if (pmtRec.GetFADCSaturatedBins(sdet::PMTConstants::eLowGain))
            station.SetLowGainSaturation();
          if (pmtRec.GetFADCSaturatedBins(sdet::PMTConstants::eHighGain))
            station.SetHighGainSaturation();
        }
      }
      return false;
    }

    int maxSignalIndex = 0;
    double maxSignal = signals[0].fCharge;
    for (unsigned int i = 1; i < nSignals; ++i) {
      if (maxSignal < signals[i].fCharge) {
        maxSignalIndex = i;
        maxSignal = signals[i].fCharge;
      }
    }

    const int start = signals[maxSignalIndex].fStart;
    const int stop = signals[maxSignalIndex].fStop;

    if (!station.HasRecData())
      station.MakeRecData();
    StationRecData& stRec = station.GetRecData();

    stRec.SetSignalStartSlot(start);
    // note that end slot is (still) inclusive
    stRec.SetSignalEndSlot(stop - 1);

    if (station.HasScintillator()) {

      auto& scintillator = station.GetScintillator();
      const auto& scintillatorSignals = scintillator.GetSignals();

      if (!scintillatorSignals.empty()) {
        // Redundant. Perhaps SignalSegmentCollection should be turned into a class
        // with functions such as GetMaxSignal that return the SignalSegment with the
        // maximum charge.
        int maxSignalScintillatorIndex = 0;
        double maxSignalScintillator = scintillatorSignals[0].fCharge;
        for (unsigned int i = 1, n = scintillatorSignals.size(); i < n; ++i) {
          if (maxSignalScintillator < scintillatorSignals[i].fCharge) {
            maxSignalScintillatorIndex = i;
            maxSignalScintillator = scintillatorSignals[i].fCharge;
          }
        }

        // start/stop considering Scintillator MIP trace only
        int scintillatorStart = scintillatorSignals[maxSignalScintillatorIndex].fStart;
        int scintillatorStop = scintillatorSignals[maxSignalScintillatorIndex].fStop;

        // If option is switched on, ensure that the integration window covers
        // the water-cherenkov detector's integration window at a minimum.
        if (fIncludeWaterCherenkovDetectorInScintillatorStartStopDetermination) {
          // time offset between WCD and Scintillator traces
          const double timeOffset = dStation.GetScintillatorPMT().GetTimeOffset();
          // floor ensures that for both negative and positive offsets, the "true" WCD
          // start time compared against the scintillator start time is within 1 bin, but
          // before the Scintillator start time.
          const int traceBinOffset = floor(timeOffset / dStation.GetFADCBinSize());
          scintillatorStart = max(0, min(scintillatorStart, start + traceBinOffset));
          scintillatorStop =
            min(int(dStation.GetFADCTraceLength()) - 1, max(scintillatorStop, stop + traceBinOffset + 1));
        }

        if (!scintillator.HasRecData())
          scintillator.MakeRecData();
        auto& scintillatorRecData = scintillator.GetRecData();

        scintillatorRecData.SetSignalStartSlot(scintillatorStart);
        scintillatorRecData.SetSignalEndSlot(scintillatorStop); // following precedent set by WCD
      } else {
        // In some cases no signals seem to be found for the Scintillator, altough they are present. 

        int scintillatorStart = 0;
        int scintillatorStop = 1;
        // If option is switched on, ensure that the integration window covers
        // the water-cherenkov detector's integration window at a minimum.
        if (fIncludeWaterCherenkovDetectorInScintillatorStartStopDetermination) {
          // time offset between WCD and Scintillator traces
          const double timeOffset = dStation.GetScintillatorPMT().GetTimeOffset();
          // floor ensures that for both negative and positive offsets, the "true" WCD
          // start time compared against the scintillator start time is within 1 bin, but
          // before the Scintillator start time.
          const int traceBinOffset = floor(timeOffset / dStation.GetFADCBinSize());
          scintillatorStart = max(0, start + traceBinOffset);
          scintillatorStop = min(int(dStation.GetFADCTraceLength()) - 1, stop + traceBinOffset + 1);
        }

        if (!scintillator.HasRecData())
          scintillator.MakeRecData();
        auto& scintillatorRecData = scintillator.GetRecData();

        scintillatorRecData.SetSignalStartSlot(scintillatorStart);
        scintillatorRecData.SetSignalEndSlot(scintillatorStop); // following precedent set by WCD
      }
    }

    {
      const auto& vemTrace = station.GetVEMTrace();
      const auto peak = for_each(&vemTrace[start+1], &vemTrace[stop], Accumulator::Max<double>(vemTrace[start]));
      stRec.SetPeakAmplitude(peak.GetMax());
    }

    bool highGainSaturation = false;
    bool lowGainSaturation = false;
    int nPMTs = 0;
    double totalCharge = 0;
    double spmtCharge = 0;
    double spmtChargeErr = 0;

    Accumulator::SampleStandardDeviationN shapeStat;
    Accumulator::SampleStandardDeviationN riseStat;
    Accumulator::SampleStandardDeviationN fallStat;
    Accumulator::SampleStandardDeviationN t50Stat;
    for (auto& pmt : station.PMTsRange(sdet::PMTConstants::eAnyType)) {

      if (pmt.HasCalibData() && pmt.GetCalibData().IsTubeOk()) {
        PMTRecData& pmtRec = pmt.GetRecData();

        if (pmt.GetType() == sdet::PMTConstants::eWaterCherenkovLarge) {
          if (pmtRec.GetFADCSaturatedBins(sdet::PMTConstants::eHighGain))
            highGainSaturation = true;
          if (pmtRec.GetFADCSaturatedBins(sdet::PMTConstants::eLowGain))
            lowGainSaturation = true;

          const TraceD& vemTrace = pmtRec.GetVEMTrace();

          double charge = accumulate(&vemTrace[start], &vemTrace[stop], 0.);

          ComputeShapeRiseFallPeak(pmtRec, dStation.GetFADCBinSize(), start, start, stop, charge);
          charge *= pmtRec.GetVEMPeak() / pmtRec.GetVEMCharge();
          pmtRec.SetTotalCharge(charge);
          totalCharge += charge;
          shapeStat(pmtRec.GetShapeParameter());
          riseStat(pmtRec.GetRiseTime());
          fallStat(pmtRec.GetFallTime());
          t50Stat(pmtRec.GetT50());
          const double peak = pmtRec.GetPeakAmplitude();
          if (peak)
            pmtRec.SetAreaOverPeak(charge / peak);
          ++nPMTs;
        } else if (pmt.GetType() == sdet::PMTConstants::eScintillator) {

          Scintillator& scintillator = station.GetScintillator();

          if (!scintillator.HasMIPTrace())
            scintillator.MakeMIPTrace();

          if (!scintillator.HasRecData())
            scintillator.MakeRecData();
          ScintillatorRecData& scinRec = scintillator.GetRecData();

          const unsigned int scintillatorStart = scinRec.GetSignalStartSlot();
          const unsigned int scintillatorStop = scinRec.GetSignalEndSlot() + 1; // following precedent set by WCD

          const TraceD& mipTrace = scintillator.GetMIPTrace();

          if (pmtRec.GetFADCSaturatedBins(sdet::PMTConstants::eHighGain))
            scintillator.SetHighGainSaturation();
          if (pmtRec.GetFADCSaturatedBins(sdet::PMTConstants::eLowGain))
            scintillator.SetLowGainSaturation();

          double charge = accumulate(&mipTrace[scintillatorStart], &mipTrace[scintillatorStop], 0.);
          ComputeShapeRiseFallPeak(pmtRec, dStation.GetFADCBinSize(), scintillatorStart, scintillatorStart, scintillatorStop, charge);
          charge *= pmtRec.GetVEMPeak() / pmtRec.GetVEMCharge();
          pmtRec.SetTotalCharge(charge);
          if (charge <= 0)
            charge = 0;
          scinRec.SetTotalSignal(charge, 0);
          scinRec.SetRiseTime(pmtRec.GetRiseTime(), 0);

        }
        else if (pmt.GetType() == sdet::PMTConstants::eWaterCherenkovSmall) {

          // Add SmallPMT saturation info to sevt::Station data
          if (pmtRec.GetFADCSaturatedBins(sdet::PMTConstants::eLowGain))
            station.SetSmallPMTSaturation();

          double peak = 0;
          double charge = 0;
          const int spmtStart = max(0, start - fBinsBeforeStartForSPMT);
          
          if(pmtRec.HasVEMTrace()){
            const TraceD& vemTrace = pmtRec.GetVEMTrace();
            charge = accumulate(&vemTrace[spmtStart], &vemTrace[stop], 0.);
            ComputeShapeRiseFallPeak(pmtRec, dStation.GetFADCBinSize(), spmtStart, spmtStart, stop, charge);
            charge *= pmtRec.GetVEMPeak() / pmtRec.GetVEMCharge();
            peak = pmtRec.GetPeakAmplitude();
          }
          spmtCharge = charge;
          spmtChargeErr = charge * pmtRec.GetVEMChargeError() / pmtRec.GetVEMCharge();
          
          pmtRec.SetTotalCharge(spmtCharge, spmtChargeErr);
          if (peak>0)
            pmtRec.SetAreaOverPeak(charge / peak);                
        }
      }
    }

    // this was done on the pmt vem traces due to individual pmt vem peak/charge values
    totalCharge /= nPMTs; // only WCD large PMTs!
    stRec.SetTotalSignal(totalCharge);
    
    if (highGainSaturation)
      station.SetHighGainSaturation();    
    if (lowGainSaturation){
      station.SetLowGainSaturation();

#if USE_SPMT_SIGNAL_AS_TOTAL
      // if LPMTs LowGain is saturated, use SmallPMT signal
      if(fIsUub && station.HasSmallPMT() &&
         station.IsSmallPMTOk()){       
        if(spmtCharge>0 && spmtChargeErr>0)
          stRec.SetTotalSignal(spmtCharge, spmtChargeErr);
        else{
          ostringstream warn;
          warn << "Station " << station.GetId() << ": zero SmallPMT signal after successful calibration!";
          WARNING(warn);
          station.SetIsSmallPMTOk(false);
        }
      }
#endif

    }

    if (totalCharge <= 0)
      return false;
    
    if (nPMTs < 2) {
      stRec.SetShapeParameter(shapeStat.GetAverage(nPMTs), 0);
      stRec.SetRiseTime(riseStat.GetAverage(nPMTs), 0);
      stRec.SetFallTime(fallStat.GetAverage(nPMTs), 0);
      stRec.SetT50(t50Stat.GetAverage(nPMTs), 0);
    } else {
      stRec.SetShapeParameter(shapeStat.GetAverage(nPMTs),
                              shapeStat.GetStandardDeviation(nPMTs));
      stRec.SetRiseTime(riseStat.GetAverage(nPMTs),
                        riseStat.GetStandardDeviation(nPMTs));
      stRec.SetFallTime(fallStat.GetAverage(nPMTs),
                        fallStat.GetStandardDeviation(nPMTs));
      stRec.SetT50(t50Stat.GetAverage(nPMTs),
                   t50Stat.GetStandardDeviation(nPMTs));
    }

    const StationGPSData& gpsData = station.GetGPSData();

    // timing of the trace END
    const TimeStamp gpsTime(gpsData.GetSecond(), gpsData.GetCorrectedNanosecond());

    const double fadcBinSize = dStation.GetFADCBinSize();

    const TimeInterval pldTimeOffset = station.GetTriggerData().GetPLDTimeOffset();

    // timing of the trace BEGINNING
    const TimeStamp traceTime = gpsTime + pldTimeOffset -
      TimeInterval(dStation.GetFADCTraceLength() * fadcBinSize);
    station.SetTraceStartTime(traceTime);

    // timing of the SIGNAL START
    const TimeStamp signalTime = traceTime + TimeInterval((start - 0.5) * fadcBinSize);
    stRec.SetSignalStartTime(signalTime);

    // timing of scintillator
    if (station.HasScintillator()) {
      Scintillator& scintillator = station.GetScintillator();
      if (scintillator.HasRecData())  {
        ScintillatorRecData& scinRec = scintillator.GetRecData();
        const TimeStamp scinSignalTime = traceTime + TimeInterval((scinRec.GetSignalStartSlot() - 0.5) * fadcBinSize);
        scinRec.SetSignalStartTime(scinSignalTime);
      }
    }

    return true;
  }

}
