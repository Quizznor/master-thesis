// Pauls stuff
#include <algorithm>
#include <numeric>
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

void DoLtpCalculation(fs::path pathToAdst)
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
        const auto pmt1 = station->GetVEMTrace(1);
        const auto pmt2 = station->GetVEMTrace(2);
        const auto pmt3 = station->GetVEMTrace(3);

        all_hits[binIndex] += 1;
        th1_hits[binIndex] += station->IsT1Threshold();
        tot_hits[binIndex] += station->IsTOT();
        totd_hits[binIndex] += station->IsTOTd();
        mops_hits[binIndex] += station->IsMoPS();

        bool IsT2Threshold = false;
        const float T2Threshold = 3.2;

        for (uint bin = 0; bin < pmt1.size(); bin++)
        {
          if (pmt1[bin] >= T2Threshold && pmt2[bin] >= T2Threshold && pmt3[bin] >= T2Threshold)
          {
            IsT2Threshold = true;
            break;
          }
        }

        th2_hits[binIndex] += IsT2Threshold;
                
      }
      else
      {
        // station was not triggered, add to "misses"
        misses[binIndex] += 1;
      }
    }

    ofstream saveFile(csvTraceFile, std::ios_base::app);
    saveFile << "0 " << log10(showerEnergy) << " " << showerZenith << " 0 0 0 0 0\n";

    for (int i = 0; i < 65; i++)
    {
      // std::cout << (i + 1) * 100 << " " << all_hits[i] << " " << misses[i] << " " << th1_hits[i] << " " << th2_hits[i] << " " << tot_hits[i] << " " << totd_hits[i] << " " << mops_hits[i] << std::endl;
      saveFile << (i + 1) * 100 << " " << all_hits[i] << " " << misses[i] << " " << th1_hits[i] << " " << th2_hits[i] << " " << tot_hits[i] << " " << totd_hits[i] << " " << mops_hits[i] << "\n";
    }

    // int allHits = std::accumulate(all_hits.begin(), all_hits.end(), 0);
    // int Misses = std::accumulate(misses.begin(), misses.end(), 0);
    // const auto equalFlag = allHits + Misses == 61 ? " == " : " != ";
    // std::cout << "Stations: " << allHits << " + " << Misses << equalFlag << allHits + Misses << std::endl;

    saveFile.close();
  }
}

int main(int argc, char** argv) 
{
  const fs::path rootDirectory{"/cr/tempdata01/filip/QGSJET-II/LTP/" + std::string(argv[1]) + "/"};
  for (const auto& file : fs::directory_iterator{rootDirectory})
  {
    DoLtpCalculation(file);
  }

  return 0;
}