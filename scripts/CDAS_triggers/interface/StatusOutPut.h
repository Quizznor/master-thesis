#ifndef _StatusOutput_
#define _StatusOutput_

#include <Rtypes.h>
#include <sd/Constants.h>
#include <vector>

typedef unsigned int uint;

struct StatusOutput
{ 
  StatusOutput() {}
  StatusOutput(const uint& gps) : fGPSSecond(gps) {}
  ~StatusOutput() {}

  const uint fGPSSecond = 0;
  //from JumpFinder (filled by DataCuts)
  std::vector<ushort> fJumpIds;
  std::vector<ushort> fUnstableIds;
  std::vector<ushort> fUnstableBaseline;
  std::vector<ushort> fUnstableScaler;

  //from DataCuts
  ushort fCutAoP = 0;
  ushort fCutTubeMask = 0;
  ushort fCutScalerRate = 0;
  ushort fCutBaseline = 0;
  ushort fCutT2 = 0;

  //Status from the Analyser (ToT type of detection, means, Active)
  ushort fnToTScaler = 0;
  ushort fnToTT2 = 0;
  ushort fnToTGlobal = 0;
  float fTriggerDensity = 0;

  float fMeanT2 = 0;
  ushort fActiveT2 = 0;
  ushort fnT2WithData = 0;
  ushort fnT2InAnalysis = 0;

  float fMeanScaler = 0;
  float fMeanScalerWoJumps = 0;
  float fMeanVarianceScaler = 0;
  ushort fActiveScaler = 0;
  ushort fnScalerWithData = 0;
  ushort fnScalerInAnalysis = 0;
  ushort fnScalerWithOldVariance = 0;

  float fDeltaFromScalerMean[sd::kNStations];

  ClassDefNV(StatusOutput, 1);
};

#endif