// Pauls stuff
#include <algorithm>

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
#include <boost/filesystem.hpp>

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
namespace fs = boost::filesystem;

struct VectorWrapper
{
  vector<float> values;

  VectorWrapper(vector<float> input_values)
  {
    values = input_values;
  }

  VectorWrapper(int size, float initial_value)
  {
    vector<float> trace_container(size, initial_value);
    values = trace_container;
  }

  VectorWrapper convert_to_VEM()
  {
    // magic number 215 = conversion factor from ADC counts to VEM equivalent
    // see David's mail from 7/06/22 for more information on conversion factor
    return this->floor() / 215.9;
  }

  VectorWrapper floor()
  {
    vector<float> result;

    for (auto x = values.begin(); x < values.end(); x++)
    {
      result.push_back(std::floor(*x));
    }

    return VectorWrapper(result);
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

  vector<float> get_trace(int start, int end)
  {
    const auto trace = std::vector<float>(values.begin() + start, values.begin() + end);
    return trace;
  }

};

void ExtractDataFromAdstFiles(fs::path pathToAdst)
{
  const auto csvTraceFile = pathToAdst.parent_path() / pathToAdst.filename().replace_extension("csv");

  // (2) start main loop
  RecEventFile     recEventFile(pathToAdst.string());
  RecEvent*        recEvent = nullptr;

  // will be assigned by root
  recEventFile.SetBuffers(&recEvent);

  // create csv file stream
  ofstream traceFile(csvTraceFile.string(), std::ios_base::app);

  // only recreate n events
  const int max_events = recEventFile.GetNEvents();
  // const int max_events = 10;

  for (unsigned int i = 0; i < max_events; ++i) 
  // for (unsigned int i = 0; i < 2; ++i) 
  {
    // skip if event reconstruction failed
    if (recEventFile.ReadEvent(i) != RecEventFile::eSuccess){continue;}

    // allocate memory for data
    const SDEvent& sdEvent = recEvent->GetSDEvent();                              // contains the traces

    // loop over all stations
    for (const auto& recStation : sdEvent.GetStationVector()) 
    {

      // only consider UUB stations
      if (!recStation.IsUUB()){continue;}
      
      // calculate shower plane distance from generated shower parameters
      const auto stationID = recStation.GetId();
      const auto showerPlaneDistance = recStation.GetSPDistance();
      const auto GPSTimestamp = recStation.GetTimeSecond();

      // #####################################################     
      // recreating trace from FADC bins

      const auto traces = recStation.GetPMTTraces();

      for (int j = 0; j < 3; j++)
      {
        const auto temp = traces[j];
        const auto trace = ( recStation.IsHighGainSaturated() ) ? temp.GetLowGainComponent() : temp.GetHighGainComponent();
        const auto baseline = ( recStation.IsHighGainSaturated() ) ? temp.GetBaselineLG() : temp.GetBaseline();
        const auto DynAnRatio = ( recStation.IsHighGainSaturated() ) ? temp.GetDynodeAnodeRatio() : 0;

        if (trace.size() != 0 && temp.GetPeak() != 0) 
        {
          traceFile << stationID << " " << showerPlaneDistance << " " << temp.GetPeak() << " " << baseline << " " << DynAnRatio << " " << GPSTimestamp << " 0 0 ";
          for (const auto& bin : trace){traceFile << bin << " ";}
          traceFile << "\n";
        }
      }

      // // #####################################################
      // // extracting the Offline reconstructed VEM trace

      // for (unsigned int PMT = 1; PMT < 4; PMT++)
      // {
      //   const auto trace = recStation.GetVEMTrace(PMT);


      //   if (trace.size() != 0) 
      //   {
      //     traceFile << stationID << " " << showerPlaneDistance << " 0 0 0 0 0 0 ";
      //     for (const auto& bin : trace){traceFile << bin << " ";}
      //     traceFile << "\n";
      //   }
      // }

      // #####################################################
    }
  }

  traceFile.close();
}

int main(int argc, char** argv) 
{
  std::cout << std::endl;
  ExtractDataFromAdstFiles(argv[1]);
  return 0;
}