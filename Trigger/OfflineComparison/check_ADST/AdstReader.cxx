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
std::vector<uint> consideredStations{

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

uint getHottestStation(const SDEvent* sdEvent)
{
  uint hottestStationId;
  Double_t maxSignal = -DBL_MAX;

  for (const auto& recStation : sdEvent->GetStationVector())
  {
    const auto depositedSignal = recStation.GetTotalSignal();

    if (depositedSignal > maxSignal)
    {
      maxSignal = depositedSignal;
      hottestStationId = recStation.GetId();
    }
  }

  return hottestStationId;
}

std::vector<uint> getCrowns(uint stationId, DetectorGeometry detectorGeometry, uint nCrowns)
{
  std::vector<uint> stationIds(1, stationId);
  std::vector<uint> lastCrown(1, stationId);

  for (auto nCrown = 1; nCrown <= nCrowns; nCrown++)
  {
    std::vector<uint> thisCrown;
    for (const auto& lastCrownId : lastCrown)
    {

      for (const auto& id : detectorGeometry.GetHexagon(lastCrownId))
      {
        const bool isConsidered= std::find(stationIds.begin(), stationIds.end(), id) != stationIds.end();

        if (!isConsidered)
        {
          stationIds.push_back(id);
          thisCrown.push_back(id);
        }
      }
    }
    lastCrown = thisCrown;
  }

  return stationIds;
}

void DoLtpCalculation(fs::path pathToAdst)
{
  std::string csvTraceFile = "/cr/tempdata01/filip/QGSJET-II/LTP/ADST/" + pathToAdst.filename().replace_extension("csv").string();

  std::cout << "writing to " + csvTraceFile << std::endl;

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
    
    std::vector<int> misses(65, 0);
    std::vector<int> all_hits(65, 0);
    std::vector<int> th1_hits(65, 0);
    std::vector<int> th2_hits(65, 0);
    std::vector<int> tot_hits(65, 0);
    std::vector<int> totd_hits(65, 0);
    std::vector<int> mops_hits(65, 0);

    // get id of all stations that received any particles (= the ones that were generated)
    std::vector<int> recreatedStationIds;
    for (const auto& recStation : sdEvent.GetStationVector()){recreatedStationIds.push_back(recStation.GetId());}

    for (const auto& consideredStationId : consideredStations)
    {
      // calculate shower plane distance
      auto showerPlaneDistance = detectorGeometry.GetStationAxisDistance(consideredStationId, showerAxis, showerCore);
      const int binIndex = floor(showerPlaneDistance / 100);

      // check if the station ID appears in the generated stations
      if (std::find(recreatedStationIds.begin(), recreatedStationIds.end(), consideredStationId) != recreatedStationIds.end())
      {
        // station was triggered, add to "hits"
        const auto station = sdEvent.GetStationById(consideredStationId);

        const auto T1ThFlag = station->IsT1Threshold();
        const auto T2ThFlag = station->IsT2Threshold();
        const auto T2ToTFlag = station->IsTOT();
        const auto T2ToTdFlag = station->IsTOTd();
        const auto T2MoPSFlag = station->IsMoPS();

        all_hits[binIndex] += 1;
        th1_hits[binIndex] += T1ThFlag;
        th2_hits[binIndex] += T2ThFlag;
        tot_hits[binIndex] += T2ToTFlag;
        totd_hits[binIndex] += T2ToTdFlag;
        mops_hits[binIndex] += T2MoPSFlag;
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
    saveFile << "0 " << log10(showerEnergy) << " " << showerZenith << " 0 0 0 0 0\n";

    for (int i = 0; i < 65; i++)
    {
      // std::cout << "<" << (i+1) * 100 << "m: " << hits[i] << " " << misses[i] << std::endl;
      saveFile << (i + 1) * 100 << " " << all_hits[i] << " " << misses[i] << " " << th1_hits[i] << " " << th2_hits[i] << " " << tot_hits[i] << " " << totd_hits[i] << " " << mops_hits[i] << "\n";
    }

    saveFile.close();
  }
}

void DoLtpCalculationQuentin(fs::path pathToAdst)
{
  std::string csvTraceFile = "/cr/tempdata01/filip/QGSJET-II/LTP/ADST/" + pathToAdst.filename().replace_extension("csv").string();

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
    
    // get id of all stations that received any particles (= the ones that were generated)
    for (const auto& station : sdEvent.GetStationVector())
    {
      const auto stationId = station.GetId();

      if (stationId >= 90000){continue;}

      // station was triggered, add to "hits"
      const auto T1ThFlag = station.IsT1Threshold();
      const auto T2ThFlag = station.IsT2Threshold();
      const auto T2ToTFlag = station.IsTOT();
      const auto T2ToTdFlag = station.IsTOTd();

      std::cout << "Station " << stationId << ": Th1: " << T1ThFlag << "; Th2 = " << T2ThFlag << "; ToT: " << T2ToTFlag << "; ToTd: " << T2ToTdFlag << std::endl;
    }
  }
}


void ExtractData(fs::path pathToAdst)
{
  // const auto csvTraceFile = pathToAdst.parent_path()/ pathToAdst.filename().replace_extension("csv"); // for testing
  const auto csvTraceFile = "/cr/tempdata01/filip/QGSJET-II/LTP/ADST_extracted/" + pathToAdst.filename().replace_extension("csv").string();

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
    ofstream traceFile(csvTraceFile, std::ios_base::app);

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


int main(int argc, char** argv) 
{
  DoLtpCalculation(argv[1]);

  return 0;
}

// /cr/tempdata01/filip/QGSJET-II/LTP/16_16.5/DAT031134_000001.root
// /cr/tempdata01/filip/QGSJET-II/LTP/16_16.5//DAT031134_000001.root