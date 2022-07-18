/**
   \file
   Implementation of CentralTriggerSimulator

   \author Darko Veberic
   \version $Id: CentralTriggerSimulator.cc 32805 2019-06-19 16:04:15Z darko $
   \date 4 May 2006
*/

#include <sdet/SDetector.h>
#include <sdet/SDetectorConstants.h>
#include <evt/Event.h>
#include <sevt/SEvent.h>
#include <sevt/SEventSimData.h>
#include <sevt/T3.h>
#include <sevt/EventTrigger.h>
#include <sevt/Header.h>
#include <sevt/Station.h>
#include <sevt/StationSimData.h>
#include <sevt/StationTriggerData.h>
#include <sevt/StationGPSData.h>
#include <sevt/PMT.h>
#include <sevt/PMTSimData.h>
#include <fevt/FEvent.h>
#include <fevt/Eye.h>
#include <fevt/EyeTriggerData.h>
#include <utl/ErrorLogger.h>
#include <utl/UTMPoint.h>
#include <utl/AugerUnits.h>
#include <utl/TabularStream.h>
#include <utl/String.h>
#include "XbAlgo.h"
#include "CentralTriggerSimulator.h"

using namespace CentralTriggerSimulatorXb;
using std::map;
using std::vector;
using std::string;
using std::ostringstream;
using namespace fwk;
using namespace utl;


// global variables for the Xb
struct Station* array = nullptr;
int nstat = 0;
int maxstat = 0;
unsigned int second_ = 0;
int newt2config = 1;
int verbose = 0;
int debug = 0;
int trivial_algo = 180;
int carmen_miranda = 300;


namespace CentralTriggerSimulatorXb {

  inline
  bool
  IsSpecialStation(const sdet::Station& station)
  {
    //return !(station.IsInGrid() || station.IsInPair()) || station.IsDense();
    return !(station.IsInGrid() ||
             station.IsInGrid(sdet::SDetectorConstants::eInfill750) ||
             station.IsInGrid(sdet::SDetectorConstants::eInfill433));
  }


  void
  MemoryCleanup()
  {
    // clear memory (ain't that ugly? what is wrong with new/delete? why malloc?)
    // memory allocated by this module
    array = nullptr;

    if (neighbour) {
      free(neighbour);
      neighbour = nullptr;
    }

    if (exists) {
      free(exists);
      exists = nullptr;
    }
  }

}


VModule::ResultFlag
CentralTriggerSimulator::Init()
{ 
  /*
   * These variables may need to be updated when there is
   * an update to XbAlgo, especially changes to the
   * reference microsecond or offset (for example, those
   * performed in the commit to r31608). The values for
   * fDefaultOffset and fDefaultWindow below are applied
   * to Dense Stations to compensate for the values
   * of offset and window from XbAlgo as applied to
   * normal (non-dense) stations.
   */
  fDefaultOffset = TimeInterval(-125*microsecond);
  fDefaultWindow = TimeInterval(30*microsecond);
  return eSuccess;
}


VModule::ResultFlag
CentralTriggerSimulator::Run(evt::Event& event)
{
  if (!event.HasSEvent())
    return eSuccess;
  sevt::SEvent& sEvent = event.GetSEvent();

  ArrayInit();
  XbAlgoT2Init();

  const sdet::SDetector& sDetector = det::Detector::GetInstance().GetSDetector();

  // collect all T2 triggers from all stations. Note that these are simulated T2s and are in StationSimData,
  // not in sevt::Station
  // Store all these triggers in a map, separating them by second.

  map<int, vector<t2>> t2InputSeconds;

  for (sevt::SEvent::ConstStationIterator sIt = sEvent.StationsBegin();
       sIt != sEvent.StationsEnd(); ++sIt) {

    if (!sIt->HasSimData())
      continue;

    try {
      const sdet::Station& dStation = sDetector.GetStation(*sIt);
      if (IsSpecialStation(dStation))
        continue;
    } catch (utl::NonExistentComponentException& ex) {
      continue;  // ignore when a station is not commissioned
    }

    const sevt::StationSimData& sSim = sIt->GetSimData();

    for (sevt::StationSimData::TriggerTimeIterator it = sSim.TriggerTimesBegin();
         it != sSim.TriggerTimesEnd(); ++it) {
      const sevt::StationTriggerData& trig = sSim.GetTriggerData(*it);
      if (!trig.IsT2())
        continue;
      const sevt::StationGPSData& gps = sSim.GetGPSData(*it);
      const int sec = gps.GetSecond();
      if (t2InputSeconds.find(sec) == t2InputSeconds.end()) {
        vector<t2> t2Input;
        t2Input.reserve(fgMaxStations);
        t2InputSeconds.insert(make_pair(sec, t2Input));
      }
      t2 t2station;
      t2station.time/*microsecond*/ = gps.GetCorrectedNanosecond() / 1000;
      t2station.ID = sIt->GetId();
      t2station.energy = trig.IsT2Threshold() ? 0 : TOTVALUE;
      t2InputSeconds[sec].push_back(t2station);
    }

  }

  // iterate over the lists of T2s (one list per second) to find posible T3s

  for (auto& st2 : t2InputSeconds) {
    second_ = st2.first;
    vector<t2>& t2Input = st2.second; // not sure if copy is needed here

    const vector<t2list> allT3Clusters = XbAlgoT2Process(&t2Input[0], t2Input.size());

    if (allT3Clusters.empty()) {
      WARNING("no T3 clusters found.");
      continue;
    }

    ostringstream info;
    info << "found " << allT3Clusters.size()
         << " T3 cluster" << String::Plural(allT3Clusters.size())
         << " while processing event second " << second_ << ':';
    INFO(info);

    // Loop over T3 clusters
    for (const auto& cluster : allT3Clusters) {

      const t3 t3Cluster = XbAlgoT3Build(cluster);

      const unsigned int refSecond = t3Cluster.refSecond;
      const int refMicroSecond = t3Cluster.refuSecond;
      const TimeStamp trigTime(refSecond, 1000*refMicroSecond);

      sevt::T3 simT3;
      simT3.SetTime(trigTime);

      const string algo = cluster.algo;
      simT3.SetAlgorithm(algo);

      info.str("");
      info << "T3 cluster: algorithm " << algo << ", "
              "reference time " << trigTime << " "
              "(" << t3Cluster.refSecond << " s, " << t3Cluster.refuSecond << " us)";
      INFO(info);

      // for T2 stations in T3 cluster
      for (unsigned int i = 0; i < cluster.nelem; ++i) {
        const t2& t3Station = cluster.point[i];
        if (!simT3.HasStation(t3Station.ID)) {
          const int dt = int(t3Station.time) - refMicroSecond;
          const TimeInterval offset = dt*microsecond;
          simT3.AddStation(t3Station.ID, offset, TimeInterval(0));
        }
      }

      // this should go into event builder...
      // Now add dense stations that are never included by Xb.
      for (sdet::SDetector::StationIterator sIt = sDetector.StationsBegin();
           sIt != sDetector.StationsEnd(); ++sIt) {
        if (IsSpecialStation(*sIt))
          simT3.AddStation(sIt->GetId(), fDefaultOffset, fDefaultWindow);
      }

      // the rest contains trigger window info (T3 stations + silents)
      const vector<t3stat>& t3Stations = t3Cluster.point;

      for (const auto& station : t3Stations) {
        if (!simT3.HasStation(station.ID))
          simT3.AddStation(station.ID, TimeInterval(station.offset*microsecond),
                           TimeInterval(station.window*microsecond));
      }

      if (!sEvent.HasSimData())
        sEvent.MakeSimData();

      sEvent.GetSimData().AddT3(simT3);

    }

    // release memory malloced by XbAlgo
    for (const auto& cluster : allT3Clusters)
      free(cluster.point);

  }

  return eSuccess;
}


VModule::ResultFlag
CentralTriggerSimulator::Finish()
{
  MemoryCleanup();
  return eSuccess;
}


void
CentralTriggerSimulator::ArrayInit()
{
  // reserve enough memory to avoid copying on resize
  fStations.clear();
  fStations.reserve(fgMaxStations);

  const sdet::SDetector& sDetector = det::Detector::GetInstance().GetSDetector();

  Station s;
  int maxId = 0;
  for (sdet::SDetector::StationIterator sIt = sDetector.StationsBegin();
       sIt != sDetector.StationsEnd(); ++sIt) {
    if (!IsSpecialStation(*sIt)) {
      const int sId = sIt->GetId();
      s.Id = sId;
      if (sId > maxId)
        maxId = sId;
      const utl::UTMPoint position(sIt->GetPosition(), utl::ReferenceEllipsoid::eWGS84);
      s.Northing = position.GetNorthing();
      s.Easting = position.GetEasting();
      s.Altitude = position.GetHeight();
      fStations.push_back(s);
    }
  }

  // add trailling station
  s.Id = -1;
  s.Northing = 0;
  s.Easting = 0;
  s.Altitude = 0;
  fStations.push_back(s);

  // fill global variables
  maxstat = maxId;
  nstat = fStations.size() - 1;
  array = &fStations[0];

  BuildNeighbour();
}
