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

  (2) start main loop
  RecEventFile     recEventFile(pathToAdst.string());
  RecEvent*        recEvent = nullptr;

  will be assigned by root
  recEventFile.SetBuffers(&recEvent);

  for (unsigned int i = 0; i < recEventFile.GetNEvents(); ++i) 
  // for (unsigned int i = 0; i < 2; ++i) 
  {
    // skip if event reconstruction failed
    if (recEventFile.ReadEvent(i) != RecEventFile::eSuccess){continue;}

    // allocate memory for data
    const SDEvent& sdEvent = recEvent->GetSDEvent();                              // contains the traces
    
    // create csv file stream
    ofstream traceFile(csvTraceFile.string(), std::ios_base::app);

    // loop over all stations
    for (const auto& recStation : sdEvent.GetStationVector()) 
    {

      // calculate shower plane distance from generated shower parameters
      const auto stationID = recStation.GetId();
      const auto showerPlaneDistance = recStation.GetSPDistance();


      // loop over all PMTs
      for (unsigned int PMT = 1; PMT < 4; PMT++)
      {
        // total trace container
        VectorWrapper TotalTrace(2048,0);

        // // get VEM trace of PMT
        // const auto TotalTrace = VectorWrapper( recStation.GetVEMTrace(PMT) );

        // loop over all components
        for (int component = ePhotonTrace; component <= eMuonTrace; component++)
        {
          const auto component_trace = recStation.GetPMTTraces((ETraceType)component, PMT);
          auto CalibratedTrace = VectorWrapper( component_trace.GetVEMComponent() );

          // make sure there exists a component of this type
          if (CalibratedTrace.values.size() != 0)
          {
            const auto vem_peak = component_trace.GetPeak();
            VectorWrapper UncalibratedTrace = CalibratedTrace * vem_peak;
            TotalTrace = TotalTrace + UncalibratedTrace;
          }
        }
        
        // write all information to trace file
        traceFile << stationID << " " << showerPlaneDistance << " 0 0";

        // "digitize" component trace...
        // this used to be converted to VEM
        const auto signal_start = recStation.GetSignalStartSlot();
        const auto signal_end = recStation.GetSignalEndSlot();
        const auto trace_vector = TotalTrace.get_trace(signal_start, signal_end);

        // ... and write to disk
        for (const auto& bin : trace_vector)
        {
          traceFile << bin << " ";
        }

        traceFile << "\n";
      }
    }

    traceFile.close();
  }
}

int main(int argc, char** argv) 
{
  std::cout << std::endl;
  ExtractDataFromAdstFiles(argv[1]);
  return 0;
}

root[] TFile f("cernstaff.root")
root[] T->StartViewer()