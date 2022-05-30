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

/**
  // PYTHON CODE SNIPPET 
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

  VectorWrapper(int size, float initial_value)
  {
    vector<float> trace_container(size, initial_value);
    values = trace_container;
  }

  VectorWrapper convert_to_VEM()
  {
    // magic number 215 = conversion factor from ADC counts to VEM equivalent
    return this->floor() / 215;
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

  vector<float> get_trace()
  {
    return values;
  }

};

void ExtractDataFromAdstFiles(fs::path pathToAdst)
{
  fs::path baseWorkingDir = "/cr/tempdata01/filip/protons";
  const auto energy = pathToAdst.lexically_relative(baseWorkingDir).remove_filename().parent_path();
  const auto csvTraceFile = pathToAdst.parent_path().parent_path() / pathToAdst.filename().replace_extension("csv");

  // (2) start main loop
  RecEventFile     recEventFile(pathToAdst.string());
  RecEvent*        recEvent = nullptr;

  // will be assigned by root
  recEventFile.SetBuffers(&recEvent);

  for (unsigned int i = 0; i < recEventFile.GetNEvents(); ++i) 
  {
    // skip if event reconstruction failed
    if (recEventFile.ReadEvent(i) != RecEventFile::eSuccess){continue;}

    // allocate memory for data
    const SDEvent& sdEvent = recEvent->GetSDEvent();
    const GenShower& genShower = recEvent->GetGenShower();

    // create csv file stream
    ofstream traceFile(csvTraceFile.string(), std::ios_base::app);

    // loop over all stations
    for (const auto& recStation : sdEvent.GetStationVector()) 
    {
      // loop over all PMTs
      for (unsigned int PMT = 1; PMT < 4; PMT++)
      {
        // total trace container
        VectorWrapper TotalTrace(2048,0);

        // loop over all components
        for (int component = ePhotonTrace; component <= eMuonTrace; component++)
        {
          const auto component_trace = recStation.GetPMTTraces((ETraceType)component, PMT);
          auto CalibratedTrace = VectorWrapper( component_trace.GetVEMComponent() );

          if (CalibratedTrace.values.size() != 0)
          {
            const auto vem_peak = component_trace.GetPeak();
            VectorWrapper UncalibratedTrace = CalibratedTrace * vem_peak;
            TotalTrace = TotalTrace + UncalibratedTrace;
          }
        }

        // get true shower parameters (energy / zenith)
        const auto showerEnergy = genShower.GetEnergy();                        // in eV
        const auto showerZenith = genShower.GetZenith() * (180 / 3.14159265);   // in °
        const auto showerPlaneDistance = recStation.GetSPDistance();            // in m

        // write all information to trace file
        traceFile << showerPlaneDistance << " " << showerEnergy << " " << showerZenith << " ";

        // "digitize" component trace...
        const auto trace_vector = TotalTrace.convert_to_VEM().get_trace();

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
  ExtractDataFromAdstFiles(argv[1]);
  return 0;
}
