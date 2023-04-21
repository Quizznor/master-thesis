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


uint CheckMoPS(fs::path pathToAdst)
{
  // (2) start main loop
  RecEventFile     recEventFile(pathToAdst.string());
  RecEvent*        recEvent = nullptr;

  // will be assigned by root
  recEventFile.SetBuffers(&recEvent);

  uint failCounter = 0;

  for (unsigned int i = 0; i < recEventFile.GetNEvents(); ++i) 
  {

    // // skip if event reconstruction failed
    // if (recEventFile.ReadEvent(i) != RecEventFile::eSuccess){continue;}

    // allocate memory for data
    const SDEvent& sdEvent = recEvent->GetSDEvent();
    const auto recStations = sdEvent.GetStationVector();

    if (recStations.size() == 3)
    {
      for (const auto& recStation : recStations)
      {
        const bool th1 = recStation.IsT1Threshold();
        const bool tot =  recStation.IsTOT();
        const bool totd = recStation.IsTOTd();
        const bool mops = recStation.IsMoPS();

        if (!th1 && !tot && !totd && mops)
        {
          std::cout << pathToAdst << std::endl;
          failCounter += 1;
        }
      }
    }
  }
  return failCounter;
}

int main(int argc, char** argv) 
{
  uint showerCounter = 0;
  uint failCounter = 0;

  const fs::path rootDirectory{"/cr/tempdata01/filip/QGSJET-II/LTP/" + std::string(argv[1]) + "/"};
  for (const auto& file : fs::directory_iterator{rootDirectory})
  {
    std::cout << "\n" << showerCounter << "\r";
    failCounter += CheckMoPS(file);
    showerCounter += 1;
  }
  std::cout << "\n" << showerCounter << " " << failCounter << std::endl;
}
