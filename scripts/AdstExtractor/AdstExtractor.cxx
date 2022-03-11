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

void ExtractDataFromAdstFiles(const string& pathToAdst, const string& pathToOutput)
{
  // (2) start main loop
  RecEventFile     recEventFile(pathToAdst);
  RecEvent*        recEvent = nullptr;

  // will be assigned by root
  recEventFile.SetBuffers(&recEvent);

  // create file for data
  const auto& start = pathToAdst.rfind('/');
  const auto& stop = pathToAdst.find_last_of('.');
  const auto& fileName = pathToAdst.substr(start, stop - start);

  for (unsigned int PMT = 1; PMT < 4; PMT++)
  {
    ofstream trace_file(pathToOutput + fileName + "_" + to_string(PMT) + ".csv");

    for (unsigned int i = 0; i < recEventFile.GetNEvents(); ++i) 
    {
      // skip if event reconstruction failed
      if (recEventFile.ReadEvent(i) != RecEventFile::eSuccess)
        continue;

      // allocate memory for data
      const SDEvent& sdEvent = recEvent->GetSDEvent();
      vector<vector<float>> traces;

      // loop over all stations
      for (const auto& recStation : sdEvent.GetStationVector()) 
      {
        // write station id to trace file
        trace_file << recStation.GetId() << ' ';
        traces.push_back(recStation.GetVEMTrace(PMT));
      }

      trace_file << endl;

      // write VEM traces to disk
      for (int j = 0; j < 2048; j++)
      {
        for (const auto &station : traces)
        {
          trace_file << station[j] << ' ';
        }
        trace_file << endl;
      }
    }
  }
}

int main(int argc, char** argv) 
{
  ExtractDataFromAdstFiles(argv[1], argv[2]);
  return 0;
}
