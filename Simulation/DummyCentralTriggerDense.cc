/**
   \file

   \author Javier Gonzalez
   \version $Id: DummyCentralTrigger.cc 23092 2013-03-20 15:02:38Z darko $
   \date 31 Mar 2022
*/

static const char CVSId[] =
  "$Id: DummyCentralTriggerDense.cc 23092 2013-03-20 15:02:38Z darko $";

#include "DummyCentralTriggerDense.h"
#include <utl/ErrorLogger.h>
#include <utl/TabularStream.h>
#include <utl/TimeStamp.h>

#include <evt/ShowerSimData.h>
#include <evt/Event.h>
#include <sevt/SEvent.h>
#include <sevt/T3.h>

#include <det/Detector.h>
#include <sdet/SDetector.h>
#include <sdet/Station.h>

using namespace sdet;
using namespace std;
using namespace sevt;
using namespace evt;
using namespace fwk;
using namespace utl;

using namespace DummyCentralTriggerDenseNS;

DummyCentralTriggerDense::DummyCentralTriggerDense(){}

DummyCentralTriggerDense::~DummyCentralTriggerDense(){}


VModule::ResultFlag
DummyCentralTriggerDense::Init()
{

  INFO("DummyCentralTriggerDense::Init()");
  fDefaultOffset = TimeInterval(0);
  fDefaultWindow = TimeInterval(1000*microsecond);
  return eSuccess;
}

VModule::ResultFlag
DummyCentralTriggerDense::Run(evt::Event & event)
{
  INFO("DummyCentralTriggerDense::Run()");

  if (!event.HasSEvent())
    return eSuccess;
  SEvent& sEvent = event.GetSEvent();

  if (!event.HasSimShower())
    return eContinueLoop;
  const ShowerSimData& simShower = event.GetSimShower();

  T3 simT3;
  TimeStamp trigTime = simShower.GetTimeStamp();
  simT3.SetTime(trigTime);
  simT3.SetAlgorithm("DummyDense");

  ostringstream info;
  info << "Dummy T3 trigger, "
    "reference time " << trigTime << " "
    "(" << trigTime.GetGPSSecond() << " s, " << int(trigTime.GetGPSNanoSecond()/1e3) << " us)" << '\n';
  INFO(info);
  TabularStream tab("r|r|l");
  tab <<              endc << "time"                       << endr
      << "station" << endc << "offset" << endc << "energy" << endr
      << hline;
  const SDetector& sDetector = det::Detector::GetInstance().GetSDetector();
  for (sdet::SDetector::StationIterator sIt = sDetector.StationsBegin();
       sIt != sDetector.StationsEnd(); ++sIt) {

    // Sufficient if we trigger all station except for dense!
    // std::vector<int> SelectedStations{
    //         5482, 5483, 5484,
    //      5439, 5440, 5441, 5442,
    //   5396, 5397, 5398, 5399, 5400,
    //      5354, 5355, 5356, 5357,
    //         5313, 5314, 5315,
    // };

    // // Trigger everything but dense stations         
    if (!(sIt->IsDense())) 
    {
      tab << sIt->GetId() << ' ' << endc
          << ' ' << fDefaultOffset << ' ' << endc
          << " !w" << fDefaultWindow << endr;
      simT3.AddStation(sIt->GetId(), fDefaultOffset, fDefaultWindow);
    }

    // // Trigger specific stations (defined in SelectedStations)
    // if (std::any_of(std::begin(SelectedStations), std::end(SelectedStations), [&](int id){return id == sIt->GetId();})) 
    // {
    //   tab << sIt->GetId() << ' ' << endc
    //       << ' ' << fDefaultOffset << ' ' << endc
    //       << " !w" << fDefaultWindow << endr;
    //   simT3.AddStation(sIt->GetId(), fDefaultOffset, fDefaultWindow);
    // }

    // // Trigger everything
    // simT3.AddStation(sIt->GetId(), fDefaultOffset, fDefaultWindow);
  }

  tab << delr;
  DEBUGLOG(tab);

  if (!sEvent.HasSimData())
    sEvent.MakeSimData();
  sEvent.GetSimData().AddT3(simT3);

  return eSuccess;
}


VModule::ResultFlag
DummyCentralTriggerDense::Finish(){
  INFO("DummyCentralTriggerDense::Finish()");
  return eSuccess;
}
