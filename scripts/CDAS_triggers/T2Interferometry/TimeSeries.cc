#include <T2Interferometry/TimeSeries.h>
#include <exception>

TimeSeries::TimeSeries() 
{
  for (uint i = 0; i < fnBins; ++i) {
    fCountStations[i].resize(250, 0);
  }
}

TimeSeries::TimeSeries(unsigned nBins) : fnBins(nBins) 
{
  for (uint i = 0; i < fnBins; ++i) {
    fCountStations[i].resize(250, 0);
  }
}

void
TimeSeries::Fill(const TimeStamp& t, unsigned bin)
{
  if (abs(t.fGPSSecond - fCurrentStartTime.fGPSSecond) > 1.) {
    throw std::out_of_range("out of time range");
    return;
  }

  if (t.fMicroSecond < fCurrentStartTime.fMicroSecond) {
    throw std::out_of_range("out of time range");
    return;
  }

  if (t.fMicroSecond > fCurrentStartTime.fMicroSecond + 250) {
    throw std::out_of_range("out of time range");
    return;
  }

  if (bin >= fnBins) {
    throw std::out_of_range("out of direction bin range");
    return;
  }

  ushort deltaMicroSecond = 0;
  if (t.fGPSSecond == fCurrentStartTime.fGPSSecond) {
    deltaMicroSecond = t.fMicroSecond - fCurrentStartTime.fMicroSecond;
  } else if (t.fGPSSecond > fCurrentStartTime.fGPSSecond) {
    deltaMicroSecond = t.fMicroSecond + 1000000 - fCurrentStartTime.fMicroSecond;
  }

  ++fCountStations[bin][deltaMicroSecond];
}

void
TimeSeries::TimeStep()
{
  fCurrentStartTime.fMicroSecond += 124;

  if (fCurrentStartTime.fMicroSecond > 1000000) {
    ++fCurrentStartTime.fGPSSecond;
    fCurrentStartTime.fMicroSecond -= 1000000;
  }
}


//assumes only one maximum
void
TimeSeries::AnalyseTimeSeries(io::RootOutFile<TimeStamp>& output)
{
  for (uint i = 0; i < fnBins; ++i) {
    ushort nMax[3] = { 0 , 0 , 0 };   // i -1, i_max, i_max + 1
    ushort jMax = 251;
    
    for (uint j = 1; j < 125; ++j) {

      if (nMax[0] + nMax[1] + nMax[2] 
          < fCountStations[i][j - 1] 
              + fCountStations[i][j] 
              + fCountStations[i][j + 1]) {
        nMax[0] = fCountStations[i][j - 1];
        nMax[1] = fCountStations[i][j];
        nMax[2] = fCountStations[i][j + 1];

        jMax = j;
      }
      fCountStations[i][j - 1] = fCountStations[i][j + 125];
      fCountStations[i][j + 125] = 0;

    }
    if (nMax[0] + nMax[1] + nMax[2] > fThreshold) {
      TimeStamp t;
      if (fCurrentStartTime.fMicroSecond + jMax < 1000000) {
        t.fGPSSecond = fCurrentStartTime.fGPSSecond;
        t.fMicroSecond = fCurrentStartTime.fMicroSecond + jMax;
      } else {
        t.fGPSSecond = fCurrentStartTime.fGPSSecond + 1;
        t.fMicroSecond = fCurrentStartTime.fMicroSecond + jMax - 1000000;
      }
     
     //t.fnStation = nMax[0] + nMax[1] + nMax[2];
     t.fDirectionBin = i;
     output << t; 
    }
  }

  TimeStep();
}