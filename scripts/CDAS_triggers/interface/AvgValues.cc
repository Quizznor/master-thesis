#include <interface/AvgValues.h>
#include <cmath>

ClassImp(AvgValues)

AvgValues::AvgValues():
  fGPSsecondBegin(0),
  fGPSsecondEnd(0)
{
  for (uint i = 0; i < sd::kNStations; ++i) {
    for (uint j = 0; j < 5; ++j) {    
      fRawMean[i][j] = 0;
      fRawVar[i][j] = 0;
      if (j < 2) 
        fActiveSeconds[i][j] = 0;
      
      if (j < 3)
        fPMTBaseline[i][j] = 0;
    }
    fMeanAoP[i] = 0;
    fJumpFlag[i] = false;
    fUnstable[i] = false;
    fUnstableScaler[i] = false;
  }

  fMeanPressure = 0;
  fMeanTemperature = 0;
}

AvgValues::AvgValues(const uint& GPSBegin, const uint& GPSEnd):
  fGPSsecondBegin(GPSBegin),
  fGPSsecondEnd(GPSEnd)
{
  for (uint i = 0; i < sd::kNStations; ++i) {
    for (uint j = 0; j < 5; ++j) {    
      fRawMean[i][j] = 0;
      fRawVar[i][j] = 0;
      if (j < 2) 
        fActiveSeconds[i][j] = 0;
      
      if (j < 3)
        fPMTBaseline[i][j] = 0;
    }
    fMeanAoP[i] = 0;
    fJumpFlag[i] = false;
    fUnstable[i] = false;
    fUnstableScaler[i] = false;
  }

  fMeanPressure = 0;
  fMeanTemperature = 0;
}


float
AvgValues::GetMean(const uint& channel, const uint& station) const
{
  if(channel > 4)
    return -1;
  if(station > sd::kNStations)
    return -1;

  return fRawMean[channel][station];
}

float
AvgValues::GetVar(const uint& channel, const uint& station) const
{
  if(channel > 4)
    return -1;
  if(station > sd::kNStations)
    return -1;

  return fRawVar[channel][station];
}

uint
AvgValues::GetBegin() const
{
  return fGPSsecondBegin;
}

uint
AvgValues::GetEnd() const
{
  return fGPSsecondEnd;
}
