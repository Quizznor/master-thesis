#ifndef _TimeSeries_
#define _TimeSeries_

#include <vector>
#include <T2Interferometry/TimeStamp.h>
#include <io/RootOutFile.h>

class TimeSeries
{
private:
  const unsigned fnBins = 120;
  TimeStamp fCurrentStartTime;    //can be greater 1e6 to compensate second boundaries

  std::vector<ushort> fCountStations[120];
  void TimeStep();    //advance by 125 micro seconds

  ushort fThreshold = 3;

public:
  TimeSeries();
  TimeSeries(unsigned nBins);
  ~TimeSeries() {}
  
  void Fill(const TimeStamp& t, unsigned bin);
  void AnalyseTimeSeries(io::RootOutFile<TimeStamp>& output);

};

#endif