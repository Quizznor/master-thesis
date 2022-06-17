#ifndef _GraphData_
#define _GraphData_

#include <Rtypes.h>

struct GraphData
{
  uint fGPSSecond = 0;

  //numbers for plotting start at 0 and run to 11 in the order as declaration here
  float fMeanCount[5];
  float fTriggerDensity = 0;
  float fMeanVarianceScaler = 0;

  uint fnToTScaler = 0;
  uint fnToTT2 = 0;
  uint fnToTGlobal = 0;

  uint fActiveScaler = 0;
  uint fActiveT2 = 0;

  uint fCutAoP = 0;
  uint fCutScalerRate = 0;
  uint fStationsWithData = 0;

  explicit operator bool() const { return fGPSSecond != 0; }
  void Reset() {(*this) = GraphData();}

  ClassDefNV(GraphData, 2);
};

#endif