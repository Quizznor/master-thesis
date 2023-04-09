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

    for (unsigned long int i = 0; i < values.size(); i++)
    {
      sum_of_both_vectors.push_back(values[i] + trace.values[i]);
    }

    return VectorWrapper(sum_of_both_vectors);
  }

  VectorWrapper operator *= (float factor)
  {
    for (auto &value : this->values){value *= factor;}

    return *this;
  }

  vector<float> get_trace(int start, int end)
  {
    // the end bin should be values.begin() + end + 1 ? Keeping this for continuity
    const auto trace = std::vector<float>(values.begin() + start, values.begin() + end);
    return trace;
  }

};

// all stations that can theoretically be triggered during simulation. Since were throwing the simulated shower anywhere near Station 5398, this 
// should ensure complete containment in most cases. Might not be true for highly inclined showers. Should in any case be a fair first estimate
std::vector<int> consideredStations{

             // 4 crowns with 5398 in center
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

    // binaries of the generated shower
    // const auto SPD = detectorGeometry.GetStationAxisDistance(Id, Axis, Core);  // in m
    const auto showerZenith = genShower.GetZenith() * (180 / 3.141593);           // in °
    const auto showerEnergy = genShower.GetEnergy();                              // in eV
    const auto showerAxis = genShower.GetAxisSiteCS();
    const auto showerCore = genShower.GetCoreSiteCS();  

    Detector detector = Detector();

    // loop over all triggered stations
    for (const auto& recStation : sdEvent.GetStationVector())
    {
      const auto stationId = recStation.GetId();
      const auto SPD = detectorGeometry.GetStationAxisDistance(stationId, showerAxis, showerCore);  // in m

      const auto genStation = sdEvent.GetSimStationById(stationId);
      const auto nMuons = genStation->GetNumberOfMuons();
      const auto nElectrons = genStation->GetNumberOfElectrons();
      const auto nPhotons = genStation->GetNumberOfPhotons();

      // Save trace in ADC/VEM format
      for (unsigned int PMT = 1; PMT < 4; PMT++)
      {

        // total trace container
        VectorWrapper TotalTrace(2048,0);

        // loop over all components (photon, electron, muons) -> NO HADRONIC COMPONENT
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
        traceFile << stationId << " " << SPD << " " << showerEnergy << " " << showerZenith << " " << nMuons << " " << nElectrons << " " << nPhotons << " ";

        // "digitize" component trace...
        // this used to be converted to VEM
        const auto signal_start = recStation.GetSignalStartSlot();
        const auto signal_end = recStation.GetSignalEndSlot();
        const auto trimmedAdcTrace = TotalTrace.get_trace(signal_start, signal_end);

        // ... and write to disk
        for (const auto& bin : trimmedAdcTrace)
        {
          traceFile << " " << bin;
        }

        traceFile << "\n";
      }
    }

    traceFile.close();
  }
}

void DoLtpCalculation(fs::path pathToAdst)
{
  std::string csvTraceFile = "/cr/tempdata01/filip/QGSJET-II/LDF/ADST/" + pathToAdst.filename().replace_extension("csv").string();

  // (2) start main loop
  RecEventFile     recEventFile(pathToAdst.string());
  RecEvent*        recEvent = nullptr;
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

    // binaries of the generated shower
    // const auto SPD = detectorGeometry.GetStationAxisDistance(Id, Axis, Core);  // in m
    const auto showerZenith = genShower.GetZenith() * (180 / 3.141593);           // in °
    const auto showerEnergy = genShower.GetEnergy();                              // in eV
    const auto showerAxis = genShower.GetAxisSiteCS();
    const auto showerCore = genShower.GetCoreSiteCS();
    
    // container for hits / misses, binned in 100m SPD
    std::vector<int> misses(65, 0);
    std::vector<int> hits(65, 0);

    // get id of all stations that triggered (for LDF, the triggers must be switched to super low threshold)
    std::vector<int> recreatedStationIds;
    for (const auto& recStation : sdEvent.GetStationVector()){recreatedStationIds.push_back(recStation.GetId());}

    for (const auto& consideredStationId : consideredStations)
    {
      // calculate shower plane distance
      auto showerPlaneDistance = detectorGeometry.GetStationAxisDistance(consideredStationId, showerAxis, showerCore);
      const int binIndex = floor(showerPlaneDistance / 100);

      // check if the station ID appears in the triggered stations
      if (std::find(recreatedStationIds.begin(), recreatedStationIds.end(), consideredStationId) != recreatedStationIds.end())
      {
        // station was triggered, add to "hits"
        hits[binIndex] += 1;
      }
      else
      {
        // station was not triggered, add to "misses"
        misses[binIndex] += 1;
      }
    }

    // save shower metadata to intermediate file. Put "0" in first 
    // column such that row 0 has the same shape as later rows
    ofstream saveFile(csvTraceFile, std::ios_base::app);
    saveFile << "0 " << log10(showerEnergy) << " " << showerZenith << "\n";

    for (int i = 0; i < 65; i++)
    {
      // std::cout << "<" << (i+1) * 100 << "m: " << hits[i] << " " << misses[i] << std::endl;
      saveFile << (i + 1) * 100 << " " << hits[i] << " " << misses[i] << "\n";
    }

    saveFile.close();

  }
}

void Compare(fs::path pathToAdst)
{
  // const auto csvTraceFile = pathToAdst.parent_path()/ pathToAdst.filename().replace_extension("csv"); // for testing
  const auto csvADCTraceFile = "/cr/users/filip/Trigger/OfflineComparison/tot_fuckups/ADC/" + pathToAdst.filename().replace_extension("csv").string();
  const auto csvVEMTraceFile = "/cr/users/filip/Trigger/OfflineComparison/tot_fuckups/VEM/" + pathToAdst.filename().replace_extension("csv").string();

  // create csv file streams
  ofstream traceADCFile(csvADCTraceFile, std::ios_base::app);
  ofstream traceVEMFile(csvVEMTraceFile, std::ios_base::app);

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

    // binaries of the generated shower
    const auto showerZenith = genShower.GetZenith() * (180 / 3.141593);           // in °
    const auto showerEnergy = genShower.GetEnergy();                              // in eV
    const auto showerAxis = genShower.GetAxisSiteCS();
    const auto showerCore = genShower.GetCoreSiteCS();

    for (const auto& recStation : sdEvent.GetStationVector())
    {
      const auto stationId = recStation.GetId();
      const auto SPD = detectorGeometry.GetStationAxisDistance(stationId, showerAxis, showerCore);  // in m

      const auto genStation = sdEvent.GetSimStationById(stationId);
      const auto nMuons = genStation->GetNumberOfMuons();
      const auto nElectrons = genStation->GetNumberOfElectrons();
      const auto nPhotons = genStation->GetNumberOfPhotons();

      // Save trace in ADC/VEM format
      for (unsigned int PMT = 1; PMT < 4; PMT++)
      {

        // total trace container
        VectorWrapper TotalTrace(2048,0);

        // loop over all components (photon, electron, muons) -> NO HADRONIC COMPONENT
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
        traceADCFile << stationId << " " << SPD << " " << showerEnergy << " " << showerZenith << " " << nMuons << " " << nElectrons << " " << nPhotons << " ";
        traceVEMFile << stationId << " " << SPD << " " << showerEnergy << " " << showerZenith << " " << nMuons << " " << nElectrons << " " << nPhotons << " ";

        // "digitize" component trace...
        // this used to be converted to VEM
        const auto signal_start = recStation.GetSignalStartSlot();
        const auto signal_end = recStation.GetSignalEndSlot();
        const auto trimmedAdcTrace = TotalTrace.get_trace(signal_start, signal_end);
        const auto VemTrace = recStation.GetVEMTrace(PMT);
        const auto trimmedVemTrace = std::vector<float>(VemTrace.begin() + signal_start, VemTrace.begin() + signal_end);

        // ... and write to disk
        for (const auto& bin : trimmedAdcTrace)
        {
          traceADCFile << " " << bin;
        }

        traceADCFile << "\n";

        for (const auto& bin : trimmedVemTrace)
        {
          traceVEMFile << " " << bin;
        }

        traceVEMFile << "\n";
      }
    }
  }

  traceADCFile.close();
  traceVEMFile.close();
}

void ExtendLowSPD(fs::path pathToAdst)
{
  // const auto csvTraceFile = pathToAdst.parent_path()/ pathToAdst.filename().replace_extension("csv"); // for testing
  const auto csvTraceFile = pathToAdst.parent_path().parent_path() / pathToAdst.filename().replace_extension("csv");
  const auto energyRange = pathToAdst.parent_path().parent_path().filename().string();

  int spdThreshold;

  switch(std::stoi(energyRange))
  {
    // catches 16-17 log(E / eV)
    case 16: 
      spdThreshold = 500;
      break;
    // catches 17-18 log(E / eV)
    case 17: 
      spdThreshold = 800;
      break;
    // catches 18-19 log(E / eV)
    case 18:
      spdThreshold = 1100;
      break;
    // catches 19-19.5 log(E / eV)
    case 19:
      spdThreshold = 1700;
      break;
  }

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

    // binaries of the generated shower
    // const auto SPD = detectorGeometry.GetStationAxisDistance(Id, Axis, Core);  // in m
    const auto showerZenith = genShower.GetZenith() * (180 / 3.141593);           // in °
    const auto showerEnergy = genShower.GetEnergy();                              // in eV
    const auto showerAxis = genShower.GetAxisSiteCS();
    const auto showerCore = genShower.GetCoreSiteCS();  

    Detector detector = Detector();

    // loop over all triggered stations
    for (const auto& recStation : sdEvent.GetStationVector())
    {
      const auto stationId = recStation.GetId();
      const auto SPD = detectorGeometry.GetStationAxisDistance(stationId, showerAxis, showerCore);  // in m
      const bool isClose = SPD < spdThreshold;

      if (!isClose){continue;}

      const auto genStation = sdEvent.GetSimStationById(stationId);
      const auto nMuons = genStation->GetNumberOfMuons();
      const auto nElectrons = genStation->GetNumberOfElectrons();
      const auto nPhotons = genStation->GetNumberOfPhotons();

      // Save trace in ADC/VEM format
      for (unsigned int PMT = 1; PMT < 4; PMT++)
      {

        // total trace container
        VectorWrapper TotalTrace(2048,0);

        // loop over all components (photon, electron, muons) -> NO HADRONIC COMPONENT
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
        traceFile << stationId << " " << SPD << " " << showerEnergy << " " << showerZenith << " " << nMuons << " " << nElectrons << " " << nPhotons << " ";
        // std::cout << stationId << " " << SPD << " " << showerEnergy << " " << showerZenith << " " << nMuons << " " << nElectrons << " " << nPhotons << std::endl;

        // "digitize" component trace...
        // this used to be converted to VEM
        const auto signal_start = recStation.GetSignalStartSlot();
        const auto signal_end = recStation.GetSignalEndSlot();
        const auto trimmedAdcTrace = TotalTrace.get_trace(signal_start, signal_end);

        // ... and write to disk
        for (const auto& bin : trimmedAdcTrace)
        {
          traceFile << " " << bin;
        }

        traceFile << "\n";
      }
    }

    traceFile.close();
  }
}

int main(int argc, char** argv) 
{
  if (std::strcmp(argv[1], "0") == 0)
  {
    // type '0' to extract traces from file
    ExtractDataFromAdstFiles(argv[2]);
  }
  else if (std::strcmp(argv[1], "1") == 0)
  {
    // type '1' for LTP/LDF fitting
    DoLtpCalculation(argv[2]);
  }
  else if (std::strcmp(argv[1], "2") == 0)
  {
    // type '2' for comparing ADC/VEM extraction
    Compare(argv[2]);
  }
    else if (std::strcmp(argv[1], "3") == 0)
  {
    // type '3' for extending dataset to low energies
    // same as '0' but only save stations w/ <1800m SPD
    ExtendLowSPD(argv[2]);
  }
  return 0;
}
