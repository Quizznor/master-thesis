// stl
#include <iostream>
#include <vector>
#include <string>
#include <cstddef>
#include <functional>
#include <set>
#include <exception>
#include <fstream>
#include <vector>

// from offline
#include <RecEventFile.h>
#include <DetectorGeometry.h>
#include <RecEvent.h>

#include <SdRecShower.h>
#include <SdRecStation.h>
#include <FdRecShower.h>
#include <FdRecStation.h>
#include <RdRecShower.h>

#include <GenShower.h>
#include <Traces.h>
#include <TraceType.h>

#include <utl/Point.h>
#include <utl/UTMPoint.h>
#include <utl/ReferenceEllipsoid.h>
#include <utl/PhysicalConstants.h>
#include <utl/AugerUnits.h>
#include <utl/AugerCoordinateSystem.h>
#include <utl/CoordinateSystem.h>
#include <utl/CoordinateSystemPtr.h>

#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_io.hpp>

using namespace std;
using namespace utl;

/**
  for iPMT in range(3):
        pmt_traces = station.GetPMTTraces(component,iPMT+1)
        calibrated_trace = np.array(pmt_traces.GetVEMComponent())
        vem_peak = pmt_traces.GetPeak()
        uncalibrated_trace = calibrated_trace*vem_peak
        if station.IsHighGainSaturated():
            dynode_anode_ratio = station.GetDynodeAnodeRatio(iPMT+1)
            uncalibrated_trace = uncalibrated_trace / dynode_anode_ratio
        traces.append(uncalibrated_trace)
    return traces
**/

struct VectorWrapper
{
  vector<float> values;

  VectorWrapper(vector<float> input_values)
  {
    values = input_values;
  }

  VectorWrapper operator * (const float factor)
  {
    vector<float> result;

    for (auto x = values.begin(); x < values.end(); x++)
    {
      result.push_back(*x * factor);
    }

    return VectorWrapper(result);
  }

  VectorWrapper operator / (const float factor)
  {
    vector<float> result;

    for (auto x = values.begin(); x < values.end(); x++)
    {
      result.push_back(*x / factor);
    }

    return VectorWrapper(result);
  }

  VectorWrapper operator + (const VectorWrapper trace)
  {

    vector<float> sum_of_both_vectors;

    for (int i = 0; i < values.size(); i++)
    {
      sum_of_both_vectors.push_back(values[i] + trace.values[i]);
    }

    return VectorWrapper(sum_of_both_vectors);
  }

  vector<float> get_trace()
  {
    return values;
  }

};

void ExtractDataFromAdstFiles(const string& pathToAdst, const string& pathToOutput)
{
  // (2) start main loop
  RecEventFile     recEventFile(pathToAdst);
  RecEvent*        recEvent = nullptr;

  // will be assigned by root
  recEventFile.SetBuffers(&recEvent);

  // create file for data
  const auto& start = pathToAdst.rfind('/');
  const auto& stop = pathToAdst.find_last_of('.');
  const auto& fileName = pathToAdst.substr(start, stop - start);

  for (unsigned int PMT = 1; PMT < 4; PMT++)
  {
    ofstream trace_file(pathToOutput + fileName + "_" + to_string(PMT) + ".csv");

    for (unsigned int i = 0; i < recEventFile.GetNEvents(); ++i) 
    {
      // skip if event reconstruction failed
      if (recEventFile.ReadEvent(i) != RecEventFile::eSuccess)
        continue;

      // allocate memory for data
      const SDEvent& sdEvent = recEvent->GetSDEvent();
      vector<vector<float>> traces;

      // loop over all stations
      for (const auto& recStation : sdEvent.GetStationVector()) 
      {
        // write station id to trace file
        trace_file << recStation.GetId() << ' ';

        // container for total trace

        const auto total_trace = recStation.GetPMTTraces(eTotalTrace, PMT);
        auto calibrated_trace = VectorWrapper( total_trace.GetVEMComponent() );

        const auto vem_peak = total_trace.GetPeak();
        VectorWrapper uncalibrated_trace = calibrated_trace * vem_peak;

        if (recStation.IsHighGainSaturated())
        {
          const auto dynode_anode_ratio = recStation.GetDynodeAnodeRatio(PMT);
          uncalibrated_trace = uncalibrated_trace / dynode_anode_ratio;
        }

        auto result = uncalibrated_trace.get_trace();
        traces.push_back(result);
      }

      trace_file << endl;

      // write VEM traces to disk
      for (int j = 0; j < 2048; j++)
      {
        for (const auto &station : traces)
        {
          trace_file << station[j] << ' ';
        }
        trace_file << endl;
      }
    }
  }
}

int main(int argc, char** argv) 
{
  ExtractDataFromAdstFiles(argv[1], argv[2]);
  return 0;
}
