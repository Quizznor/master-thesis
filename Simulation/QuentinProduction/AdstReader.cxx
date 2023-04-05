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

  const auto energyRange = pathToAdst.parent_path().parent_path().filename().string();
  std::string csvTraceFile = "/cr/tempdata01/filip/QGSJET-II/QUENTIN/" + energyRange + "/" + pathToAdst.filename().replace_extension("csv").string();

  std::cout << csvTraceFile << std::endl;

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

    // get all stations within 4 crowns of the hottest station
    const auto consideredStations = getCrowns(getHottestStation(&sdEvent) , detectorGeometry, 4);

    // binaries of the generated shower
    // const auto SPD = detectorGeometry.GetStationAxisDistance(Id, Axis, Core);  // in m
    const auto showerZenith = genShower.GetZenith() * (180 / 3.141593);           // in °
    const auto showerEnergy = genShower.GetEnergy();                              // in eV
    const auto showerAxis = genShower.GetAxisSiteCS();
    const auto showerCore = genShower.GetCoreSiteCS();
    
    std::vector<int> misses(65, 0);
    std::vector<int> all_hits(65, 0);
    std::vector<int> th_hits(65, 0);
    std::vector<int> tot_hits(65, 0);
    std::vector<int> totd_hits(65, 0);

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

        all_hits[binIndex] += 1;
        th_hits[binIndex] += station->IsT2Threshold();
        tot_hits[binIndex] += station->IsTOT();
        totd_hits[binIndex] += station->IsTOTd();
      }
      else
      {
        // station was not triggered, add to "misses"
        misses[binIndex] += 1;
      }
    }

    ofstream saveFile(csvTraceFile, std::ios_base::app);
    saveFile << "0 " << log10(showerEnergy) << " " << showerZenith << " 0 0 0\n";

    for (int i = 0; i < 65; i++)
    {
      // std::cout << "<" << (i+1) * 100 << "m: " << hits[i] << " " << misses[i] << std::endl;
      saveFile << (i + 1) * 100 << " " << all_hits[i] << " " << misses[i] << " " << th_hits[i] << " " << tot_hits[i] << " " << totd_hits[i] << "\n";
    }

    saveFile.close();
  }
}

int main(int argc, char** argv) 
{
  DoLtpCalculation(argv[1]);
  return 0;
}
