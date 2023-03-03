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

void doCrossCheck(fs::path pathToAdst)
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
        const auto stationPosition = detectorGeometry.GetStationPosition(consideredStationId);

        // calculate shower plane distance for considered station
        const auto showerPlaneDistanceById = detectorGeometry.GetStationAxisDistance(consideredStationId, showerAxis, showerCore);
        const auto showerPlaneDistanceByPos = detectorGeometry.GetStationAxisDistance(stationPosition, showerAxis, showerCore);

        std::cout << consideredStationId << ": " << showerPlaneDistanceById << " <=> " << showerPlaneDistanceByPos << std::endl;
    }
  }
}

int main(int argc, char** argv) 
{
  doCrossCheck(argv[1]);
  return 0;
}