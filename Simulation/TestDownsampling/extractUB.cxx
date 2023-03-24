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

std::vector<double> multiply(std::vector<short unsigned int> vector, double factor)
{
  std::vector<double> result;
  for (const auto& item : vector){result.push_back(item * factor);}
  return result;
}

void ExtractUB(fs::path pathToAdst)
{
  // const auto csvTraceFile = pathToAdst.parent_path()/ pathToAdst.filename().replace_extension("csv"); // for testing
  const auto csvTraceFile = pathToAdst.parent_path().parent_path() / pathToAdst.filename().replace_extension("csv");

  std::cout << csvTraceFile << std::endl;

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
      for (int i_PMT = 1; i_PMT < 4; i_PMT++)
      {
        // write all information to trace file
        traceFile << stationId << " " << SPD << " " << showerEnergy << " " << showerZenith << " " << nMuons << " " << nElectrons << " " << nPhotons << " ";
        
        const auto signal_start = recStation.GetSignalStartSlot();
        const auto signal_end = recStation.GetSignalEndSlot();

        const auto trace = recStation.GetVEMTrace(i_PMT);

        // the end bin should be values.begin() + end + 1 ? Keeping this for continuity
        const auto traceSliced = std::vector<float>(trace.begin() + signal_start, trace.begin() + signal_end + 1);

        for (const auto& bin : traceSliced){traceFile << " " << bin;}

        traceFile << "\n";
      }
    }

    // traceFile.close();
  }
}

int main(int argc, char** argv) 
{
  ExtractUB(argv[1]);
  return 0;
}
