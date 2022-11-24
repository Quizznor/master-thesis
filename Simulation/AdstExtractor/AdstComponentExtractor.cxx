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

// all stations that can theoretically be triggered during simulation. Since were throwing the simulated shower anywhere near Station 5398, this 
// should ensure complete containment in most cases. Might not be true for highly inclined showers. Should in any case be a fair first estimate
std::vector<int> consideredStations{

             // 4 rings with 5398 in center
              4049, 4050, 4051, 4052, 4053,
            4006, 4007, 4008, 4009, 4010, 4011,
        5480, 5481, 5482, 5483, 5484, 5485, 5486,
      5437, 5438, 5439, 5440, 5441, 5442, 5443, 5444,
  5394, 5395, 5396, 5397, 5398, 5399, 5400, 5401, 5402,
      5352, 5353, 5354, 5355, 5356, 5357, 5358, 5359,
        5311, 5312, 5313, 5314, 5315, 5316, 5317,
            5270, 5271, 5272, 5273, 5274, 5275,
              5230, 5231, 5232, 5233, 5234
};

void ExtractDataFromAdstFiles(fs::path pathToAdst)
{
  // const auto csvTraceFile = pathToAdst.parent_path()/ pathToAdst.filename().replace_extension("csv"); // for testing
  const auto csvTraceFile = pathToAdst.parent_path().parent_path() / pathToAdst.filename().replace_extension("csv");
  const std::string csvLdfFileMisses = "/cr/tempdata01/filip/QGSJET-II/lateral_density_function/misses.csv";
  const std::string csvLdfFileHits = "/cr/tempdata01/filip/QGSJET-II/lateral_density_function/hits.csv";

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
    const SDEvent& sdEvent = recEvent->GetSDEvent();                              // contains the traces
    const GenShower& genShower = recEvent->GetGenShower();                        // contains the shower
    DetectorGeometry detectorGeometry = DetectorGeometry();                       // contains SPDistance
    recEventFile.ReadDetectorGeometry(detectorGeometry);

    // create csv file streams
    ofstream traceFile(csvTraceFile.string(), std::ios_base::app);
    ofstream ldfFileMisses(csvLdfFileMisses, std::ios_base::app);
    ofstream ldfFileHits(csvLdfFileHits, std::ios_base::app);

    // binaries of the generated shower
    // const auto SPD = detectorGeometry.GetStationAxisDistance(Id, Axis, Core);  // in m
    const auto showerZenith = genShower.GetZenith() * (180 / 3.141593);           // in °
    const auto showerEnergy = genShower.GetEnergy();                              // in eV
    const auto showerAxis = genShower.GetAxisSiteCS();
    const auto showerCore = genShower.GetCoreSiteCS();  

    // get id of all stations that participated in trigger
    // get no. of particles that received any particles
    std::vector<int> recreatedStationIds;
    std::vector<int> simulatedStationIds;

    for (const auto& recStation : sdEvent.GetStationVector()){recreatedStationIds.push_back(recStation.GetId());}
    for (const auto& genStation : sdEvent.GetSimStationVector()){simulatedStationIds.push_back(genStation.GetId());}

    Detector detector = Detector();

    // loop over all considered Stations
    for (const auto& consideredStationId : consideredStations)
    {

      const auto genIndex = std::find(simulatedStationIds.begin(), simulatedStationIds.end(), consideredStationId);

      // calculate shower plane distance for considered station
      const auto showerPlaneDistance = detectorGeometry.GetStationAxisDistance(consideredStationId, showerAxis, showerCore);

      // check if considered station has received particles
      if (genIndex != simulatedStationIds.end())
      {
        // simulated station is hit by the shower
        //      -> write spd and theta to hits.csv
        //      -> check if it actually triggered

        const auto genIndex = std::find(simulatedStationIds.begin(), simulatedStationIds.end(), consideredStationId);
        const auto stationId = simulatedStationIds[genIndex - simulatedStationIds.begin()];
        const auto genStation = sdEvent.GetSimStationById(stationId);

        const auto nMuons = genStation->GetNumberOfMuons();
        const auto nElectrons = genStation->GetNumberOfElectrons();
        const auto nPhotons = genStation->GetNumberOfPhotons();

        ldfFileHits << showerPlaneDistance << " " << showerEnergy << " " << showerZenith << " " << nMuons << ", " << nElectrons << ", " << nPhotons << std::endl;

        if (std::find(recreatedStationIds.begin(), recreatedStationIds.end(), consideredStationId) != recreatedStationIds.end())
        {
          // station participated in the trigger
          //    -> get injected particles from GenStation
          //    -> do the rest of the SD reconstruction

          // fetch data from desired station
          const auto recStation = sdEvent.GetStationById(stationId);

          std::cout << "Station " << stationId << " at SPD = " << showerPlaneDistance << "m received (" << nMuons << ", " << nElectrons << ", " << nPhotons << ") particles" << std::endl;

          // loop over all PMTs
          for (unsigned int PMT = 1; PMT < 4; PMT++)
          {
            // total trace container
            VectorWrapper TotalTrace(2048,0);

            // loop over all components (photon, electron, muons) -> NO HADRONIC COMPONENT
            for (int component = ePhotonTrace; component <= eMuonTrace; component++)
            {
              const auto component_trace = recStation->GetPMTTraces((ETraceType)component, PMT);
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
            traceFile << stationId << " " << showerPlaneDistance << " " << showerEnergy << " " << showerZenith << " " << nMuons << " " << nElectrons << " " << nPhotons;

            // "digitize" component trace...
            // this used to be converted to VEM
            const auto signal_start = recStation->GetSignalStartSlot();
            const auto signal_end = recStation->GetSignalEndSlot();
            const auto trace_vector = TotalTrace.get_trace(signal_start, signal_end);

            // ... and write to disk
            for (const auto& bin : trace_vector)
            {
              traceFile << " " << bin;
            }

            traceFile << "\n";
          }
        }
      }
      else
      {
        // simulated station is not hit by the shower
        //   -> write spd and theta to misses.csv

        ldfFileMisses << showerPlaneDistance << " " << showerEnergy << " " << showerZenith << " 0 0 0" << std::endl;
      }
    }

    traceFile.close();
    ldfFileHits.close();
    ldfFileMisses.close();
  }
}

int main(int argc, char** argv) 
{
  std::cout << std::endl;
  ExtractDataFromAdstFiles(argv[1]);
  return 0;
}
