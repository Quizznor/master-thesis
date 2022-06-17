#include <T2Dump/MCGenerator.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <t2/T2Data.h>
#include <t2/T2DumpFile.h>


/*void
MCGenerator::ReadPositions(const std::string& filename)
{
  std::ifstream inPositions(filename);
  double x = 0;
  double y = 0;
  double z = 0;
  int id = 0;

  double tmp = 0;
   
  //read in converted to micro seconds
  while (inPositions >> id >> y >> x >> z >> tmp >> tmp >> tmp >> tmp >> tmp >> tmp) {
    fPositions[id].SetXYZ(x/300.,
                          y/300.,
                          z/300.);
  }
}*/


void
MCGenerator::GetSingleT2Id(T2& t2, std::mt19937_64& rand)
{
  std::uniform_int_distribution<std::mt19937_64::result_type> dist(0, 1999);
  while (true) {
    int id = dist(rand);
    if (fStationMask[id]) {
      t2.fId = id;
      if (t2.fId > 1999)
        std::cerr << "warning: using fake stations!" << std::endl;
      return;
    }
  }
}


double 
MCGenerator::GetMinimalDistance(const std::vector<ushort>& ids, ushort id) 
  const
{
  double minDistance = 1e9;
  for (const auto& i : ids) {
    const double distance2 = fStationInfo[id].fPosition.Distance2(fStationInfo[i].fPosition);
    if (distance2 < minDistance) 
      minDistance = distance2;
  }

  return std::sqrt(minDistance);
}

void
MCGenerator::CreateEvent(std::mt19937_64& rand,
                         const TVector3& direction,
                         int microSecond)
{
  std::uniform_int_distribution<std::mt19937_64::result_type> nStationsDist(3, fMaximalNumberOfStationsPerEvent);
  CreateEvent(rand, direction, microSecond, nStationsDist(rand));
}


void
MCGenerator::CreateEvent(std::mt19937_64& rand,
            const TVector3& direction,
            int microSecond,
            uint nStations)
{
  //const TVector3 referencePoint(473872./300., 6104846./300., 1500./300.);
  std::uniform_int_distribution<std::mt19937_64::result_type> dist(0, 1999);

  MCEvent mcEvent;
  mcEvent.fGPSSecond = fGPSSecond;
  mcEvent.fMicroSecond = microSecond;

  while (mcEvent.fIds.size() < nStations) {
    int id = dist(rand);
    if (!fStationMask[id])
      continue;
    if (!std::count(mcEvent.fIds.begin(), mcEvent.fIds.end(), id)) {
      mcEvent.fIds.push_back(id);
      mcEvent.fPositions.emplace_back(fStationInfo[id].fPosition.fX, 
                                      fStationInfo[id].fPosition.fY);
    }
  }

  for (const auto& id : mcEvent.fIds) {
    int deltaT = - ( direction.x()*fStationInfo[id].fPosition.fX 
                 + direction.y()*fStationInfo[id].fPosition.fY 
                 + direction.z()*fStationInfo[id].fPosition.fZ)/300.;
    if (abs(fLastTriggerTime[id] - mcEvent.fMicroSecond + deltaT) < 19)
      continue;
    fLastTriggerTime[id] = mcEvent.fMicroSecond + deltaT;
    foutput[fcurrentIndex++] = T2(mcEvent.fMicroSecond + deltaT, 
                                  id, fEventTriggerType(rand));
    
    mcEvent.fTimes.push_back(mcEvent.fMicroSecond + deltaT);
  }

  mcEvent.fTheta = direction.Theta();
  mcEvent.fPhi = direction.Phi();

  mcEvent.fCompact = false;
  mcEvent.fType = 1;

  fOutput << mcEvent;
}

void
MCGenerator::CreateSpallationEvent(std::mt19937_64& rand, 
                                   const TVector3& direction,
                                   int microSecond)
{
  std::uniform_int_distribution<std::mt19937_64::result_type> nStationsDist(3, fMaximalNumberOfStationsPerEvent);
  CreateSpallationEvent(rand, direction, microSecond, nStationsDist(rand)); 
}

void
MCGenerator::CreateSpallationEvent(std::mt19937_64& rand, 
                                   const TVector3& direction,
                                   int microSecond,
                                   uint nStations)
{
  const int probabilityForNewSubShower = 2; /* *2e-3 */
  std::uniform_int_distribution<std::mt19937_64::result_type> dist(0, 1999);
  std::uniform_int_distribution<std::mt19937_64::result_type> distBegin(150, 999750);
  
  MCEvent mcEvent;
  mcEvent.fGPSSecond = fGPSSecond;

  int n = 0;

  while (mcEvent.fIds.size() < nStations) {
    int id = dist(rand);
    ++n;
    if (n > 100000)
      break;

    if (!fStationMask[id])
      continue;    
    
    if (!mcEvent.fIds.size()) {
      mcEvent.fIds.push_back(id);
      mcEvent.fPositions.emplace_back(fStationInfo[id].fPosition.fX, 
                                      fStationInfo[id].fPosition.fY);
    } else if (std::count(mcEvent.fIds.begin(), mcEvent.fIds.end(), id)) {
      continue;
    } else {
      double minDistance = GetMinimalDistance(mcEvent.fIds, id);
      const double distToFirst = std::sqrt(fStationInfo[id].fPosition.Distance2(fStationInfo[mcEvent.fIds.front()].fPosition));
      if ((minDistance < 1600. && distToFirst < 10000.)
         || minDistance < 1600.
         || dist(rand) < probabilityForNewSubShower) {
        mcEvent.fIds.push_back(id);
        mcEvent.fPositions.emplace_back(fStationInfo[id].fPosition.fX, 
                                        fStationInfo[id].fPosition.fY);
      }
    }
  }
  mcEvent.fMicroSecond = microSecond;

  for (const auto& id : mcEvent.fIds) {
    int deltaT = - floor( direction.x()*fStationInfo[id].fPosition.fX 
                   + direction.y()*fStationInfo[id].fPosition.fY 
                   + direction.z()*fStationInfo[id].fPosition.fZ)/300.;
    if (abs(fLastTriggerTime[id] - mcEvent.fMicroSecond + deltaT) < 19)
      continue;
    fLastTriggerTime[id] = mcEvent.fMicroSecond + deltaT;
    foutput[fcurrentIndex++] = T2(mcEvent.fMicroSecond + deltaT, id, fEventTriggerType(rand));
    mcEvent.fTimes.push_back(mcEvent.fMicroSecond + deltaT);
  }

  mcEvent.fTheta = direction.Theta();
  mcEvent.fPhi = direction.Phi();
  mcEvent.fType = 2;

  fOutput << mcEvent;
}


void
MCGenerator::CreateRingEvent(std::mt19937_64& rand, int microSecond)
{
  std::uniform_int_distribution<std::mt19937_64::result_type> nStationsDist(7, fMaximalNumberOfStationsPerEvent);
  CreateRingEvent(rand, microSecond, nStationsDist(rand));
}


void
MCGenerator::CreateRingEvent(std::mt19937_64& rand,
                             int microSecond, 
                             uint nStations)
{
  std::uniform_int_distribution<std::mt19937_64::result_type> dist(0, 1999);
  std::uniform_int_distribution<std::mt19937_64::result_type> distHeight(2000, 7500);

  MCEvent mcEvent;
  mcEvent.fGPSSecond = fGPSSecond;
  mcEvent.fPhi = distHeight(rand);

  int n = 0;

  while (mcEvent.fIds.size() < nStations) {
    int id = dist(rand);
    ++n;
    if (n > 10000)
      break;

    if (!fStationMask[id])
      continue;    
    
    if (!mcEvent.fIds.size()) {
      mcEvent.fIds.push_back(id);
      mcEvent.fPositions.emplace_back(fStationInfo[id].fPosition.fX, 
                                      fStationInfo[id].fPosition.fY);
      n = 0;
    } else if (std::count(mcEvent.fIds.begin(), mcEvent.fIds.end(), id)) {
      continue;
    } else {
      const double minDistance = GetMinimalDistance(mcEvent.fIds, id);
      if (minDistance < 4000.) {
        mcEvent.fIds.push_back(id);
        mcEvent.fPositions.emplace_back(fStationInfo[id].fPosition.fX, 
                                        fStationInfo[id].fPosition.fY);
        n = 0;
      }
    }
  }
  mcEvent.fMicroSecond = microSecond;
  const t2::Vector<double> centerPosition(fStationInfo[mcEvent.fIds.front()].fPosition.fX 
                                            + dist(rand) - 1000.,
                                          fStationInfo[mcEvent.fIds.front()].fPosition.fY
                                            + dist(rand) - 1000.,
                                          fStationInfo[mcEvent.fIds.front()].fPosition.fZ 
                                           + mcEvent.fPhi);   
  const t2::Vector<double> centerOnGround(centerPosition.fX, centerPosition.fY, 0);

  for (const auto& id : mcEvent.fIds) {
    int deltaT = std::sqrt(fStationInfo[id].fPosition.Distance2(centerPosition))/300.; 
    if (fStationInfo[id].fPosition.Distance2(centerOnGround) < 1500.*1500.)
      deltaT -= 20.; //shift innermost triggers as seen in real data
    //not using fEventTriggerType as lightning stations are assumed to be
    // 'long signal' stations -> ToT
    if (abs(fLastTriggerTime[id] - mcEvent.fMicroSecond + deltaT) < 19)
      continue;
    fLastTriggerTime[id] = mcEvent.fMicroSecond + deltaT;

    foutput[fcurrentIndex++] = T2(mcEvent.fMicroSecond + deltaT, id); 
    mcEvent.fTimes.push_back(mcEvent.fMicroSecond + deltaT);
  }

  mcEvent.fTheta = 0;
  mcEvent.fType = 3;

  fOutput << mcEvent;
}


void 
MCGenerator::CreateHorizontalShower(std::mt19937_64& rand, 
                                    const TVector3& direction,
                                    int microSecond)
{
  std::uniform_int_distribution<std::mt19937_64::result_type> nStationsDist(5, fMaximalNumberOfStationsPerEvent);
  CreateHorizontalShower(rand, direction, microSecond, nStationsDist(rand));
}


void
MCGenerator::CreateHorizontalShower(std::mt19937_64& rand, 
                                    const TVector3& direction,
                                    int microSecond,
                                    uint nStations)
{
  std::uniform_int_distribution<std::mt19937_64::result_type> dist(0, 1999);
  std::uniform_real_distribution<> distCore(-15000, 15000);

  MCEvent mcEvent;
  mcEvent.fGPSSecond = fGPSSecond;
  mcEvent.fMicroSecond = microSecond;

  mcEvent.fTheta = direction.Theta();
  mcEvent.fPhi = direction.Phi();

  const t2::Vector<double> core(distCore(rand), distCore(rand), 0);

  for (int i = 0; i < 2000; ++i) {
    if (!fStationMask[i])
      continue;

    if (mcEvent.fIds.size() == nStations)
      break;

    if (!mcEvent.fIds.size()) {
      if (GetDistanceToLine(fStationInfo[i].fPosition, mcEvent.fPhi, core) < 1000.) {
        mcEvent.fIds.push_back(i);
      }
    } else {
      const double lineDistance = GetDistanceToLine(fStationInfo[i].fPosition, mcEvent.fPhi, core);
      const double minDistance = GetMinimalDistance(mcEvent.fIds, i);
      
      if (lineDistance < 750. && minDistance < 4500) {
        mcEvent.fIds.push_back(i);
      }      
    }
  }  

  for (const auto& id : mcEvent.fIds) {
    int deltaT = - ( direction.x()*fStationInfo[id].fPosition.fX 
                   + direction.y()*fStationInfo[id].fPosition.fY 
                   + direction.z()*fStationInfo[id].fPosition.fZ)/300.;
    if (abs(fLastTriggerTime[id] - mcEvent.fMicroSecond + deltaT) < 19)
      continue;
    fLastTriggerTime[id] = mcEvent.fMicroSecond + deltaT;

    foutput[fcurrentIndex++] = T2(mcEvent.fMicroSecond + deltaT, id, fEventTriggerType(rand));
    mcEvent.fTimes.push_back(mcEvent.fMicroSecond + deltaT);
    mcEvent.fPositions.emplace_back(fStationInfo[id].fPosition.fX, 
                                        fStationInfo[id].fPosition.fY);
  }  

  mcEvent.fType = 4;

  fOutput << mcEvent;
}


void
MCGenerator::CreateShower(std::mt19937_64& rand, const TVector3& direction, int microSecond)
{
  std::uniform_int_distribution<std::mt19937_64::result_type> nStationsDist(2, fMaximalNumberOfStationsPerEvent);
  CreateShower(rand, direction, microSecond, nStationsDist(rand));
}


void
MCGenerator::CreateShower(std::mt19937_64& rand, 
                          const TVector3& direction, 
                          int microSecond,
                          uint nStations)
{
  std::uniform_int_distribution<std::mt19937_64::result_type> dist(0, 1999);
  
  MCEvent mcEvent;
  mcEvent.fGPSSecond = fGPSSecond;

  int n = 0;

  while (mcEvent.fIds.size() < nStations) {
    int id = dist(rand);
    ++n;
    if (n > 1000000)
      break;

    if (!fStationMask[id])
      continue;    
    
    if (!mcEvent.fIds.size()) {
      mcEvent.fIds.push_back(id);
      mcEvent.fPositions.emplace_back(fStationInfo[id].fPosition.fX, 
                                      fStationInfo[id].fPosition.fY);
    } else if (std::count(mcEvent.fIds.begin(), mcEvent.fIds.end(), id)) {
      continue;
    } else {
      double minDistance = GetMinimalDistance(mcEvent.fIds, id);
      const double distToFirst = std::sqrt(fStationInfo[id].fPosition.Distance2(fStationInfo[mcEvent.fIds.front()].fPosition));
      if (minDistance < 1600. && distToFirst < 15000.) {
        mcEvent.fIds.push_back(id);
        mcEvent.fPositions.emplace_back(fStationInfo[id].fPosition.fX, 
                                        fStationInfo[id].fPosition.fY);
      }
    }
  }
  mcEvent.fMicroSecond = microSecond;

  for (const auto& id : mcEvent.fIds) {
    int deltaT = - ( direction.x()*fStationInfo[id].fPosition.fX 
                   + direction.y()*fStationInfo[id].fPosition.fY 
                   + direction.z()*fStationInfo[id].fPosition.fZ)/300.;
    if (abs(fLastTriggerTime[id] - mcEvent.fMicroSecond + deltaT) < 19)
      continue;
    fLastTriggerTime[id] = mcEvent.fMicroSecond + deltaT;
    foutput[fcurrentIndex++] = T2(mcEvent.fMicroSecond + deltaT, id, fEventTriggerType(rand));
    mcEvent.fTimes.push_back(mcEvent.fMicroSecond + deltaT);
  }

  mcEvent.fTheta = direction.Theta();
  mcEvent.fPhi = direction.Phi();

  mcEvent.fType = 0;
  mcEvent.fCompact = true;

  fOutput << mcEvent;
}

void
MCGenerator::SetCosTheta(double cth)
{
  if (cth > 1 || cth < 0)
    throw std::invalid_argument("cosine theta has to be in [0, 1)");
  fCosTheta = cth; 
}


void 
MCGenerator::EventGeneration(std::mt19937_64& random, 
                             int microSecond, 
                             uint nStations = 0)
{
  short type = fType;
  if (type == -1) {
    std::uniform_int_distribution<std::mt19937_64::result_type> dist(0, 4);
    type = dist(random);
  }

  TVector3 axis;
  axis.SetXYZ(1., 0, 0);
  
  std::uniform_real_distribution<> distPhi(0, 2*3.1415);
  axis.SetPhi(distPhi(random));

  if (fCosTheta < 0) {
    std::uniform_real_distribution<> distTheta(0.5, 1); 
    axis.SetTheta(acos(distTheta(random)));
  } else {
    axis.SetTheta(acos(fCosTheta));
  }

  switch (type) {
    case 0:
      if (!nStations)
        CreateShower(random, axis, microSecond);
      else
        CreateShower(random, axis, microSecond, nStations);
    break;

    case 1:
      if (!nStations)
        CreateEvent(random, axis, microSecond);
      else
        CreateEvent(random, axis, microSecond, nStations);
    break;

    case 2:
      if (!nStations)
        CreateSpallationEvent(random, axis, microSecond);
      else 
        CreateSpallationEvent(random, axis, microSecond, nStations);
    break;

    case 3:
      if (!nStations)
        CreateRingEvent(random, microSecond);
      else
        CreateRingEvent(random, microSecond, nStations);
    break;

    case 4: 
    {
      TVector3 horizontalAxis(axis);
      std::uniform_real_distribution<> distTheta(0., 0.5); 
      horizontalAxis.SetTheta(acos(distTheta(random)));
      if (!nStations)
        CreateHorizontalShower(random, horizontalAxis, microSecond);
      else
        CreateHorizontalShower(random, horizontalAxis, microSecond, nStations);
    }
    break;

    default:
      std::cerr << "Unknown event type: this should never happen!" << std::endl;
  } 
}

void 
MCGenerator::EventGeneration(std::mt19937_64& random, 
                             int microSecond, 
                             std::discrete_distribution<>& p)
{
  int key = p(random);
  short type = key / 100;
  int nStations = key % 100;

  TVector3 axis;
  axis.SetXYZ(1., 0, 0);
  
  std::uniform_real_distribution<> distPhi(0, 2*3.1415);
  axis.SetPhi(distPhi(random));

  if (fCosTheta < 0) {
    std::uniform_real_distribution<> distTheta(0.5, 1); 
    axis.SetTheta(acos(distTheta(random)));
  } else {
    axis.SetTheta(acos(fCosTheta));
  }

  switch (type) {
    case 0:
      if (!nStations)
        CreateShower(random, axis, microSecond);
      else
        CreateShower(random, axis, microSecond, nStations);
    break;

    case 1:
      if (!nStations)
        CreateEvent(random, axis, microSecond);
      else
        CreateEvent(random, axis, microSecond, nStations);
    break;

    case 2:
      if (!nStations)
        CreateSpallationEvent(random, axis, microSecond);
      else 
        CreateSpallationEvent(random, axis, microSecond, nStations);
    break;

    case 3:
      if (!nStations)
        CreateRingEvent(random, microSecond);
      else
        CreateRingEvent(random, microSecond, nStations);
    break;

    case 4: 
    {
      TVector3 horizontalAxis(axis);
      std::uniform_real_distribution<> distTheta(0., 0.5); 
      horizontalAxis.SetTheta(acos(distTheta(random)));
      if (!nStations)
        CreateHorizontalShower(random, horizontalAxis, microSecond);
      else
        CreateHorizontalShower(random, horizontalAxis, microSecond, nStations);
    }
    break;

    default:
      std::cerr << "Unknown event type: this should never happen!" << std::endl;
  } 
}

void
MCGenerator::GenerateT2(int n, 
                        const std::string& outfilename, 
                        int nPerSec,
                        int nStationsPerEvent = 0)
{
  std::ofstream outStream(outfilename, std::ios::binary);
  std::vector<double> weights;
  
  ReadConfigFile(weights);
  std::discrete_distribution<> eventTypeDistribution(weights.begin(), weights.end());

  double newEventTime = 0;

  std::mt19937_64 random(fSeed);
  if (!fSeed)
    random.seed(std::random_device()());
  std::exponential_distribution<double> expDist(nPerSec/1e6);
  std::uniform_int_distribution<std::mt19937_64::result_type> dist(0, 100); //type of T2
  std::exponential_distribution<double> eventDist(fEventRate/1e6);
  
  newEventTime = eventDist(random);

  if (!fEventRate)
    newEventTime = (n+1)*1e6;
  
  for (int i = 0; i < n; ++i) {
    if (!(i % 300))
      std::cout << i << " out of " << n << std::endl;

    outStream.write((char*)&fGPSSecond, sizeof(fGPSSecond));

    fcurrentIndex = 0;

    while (fcurrentMicroS < 1000000) {
      T2 t2;
      const double timeStep = expDist(random);
      fcurrentMicroS += timeStep;
      newEventTime -= timeStep;

      if (newEventTime <= 0) {
        ++fEventCounter;
        if (fcurrentMicroS + newEventTime < 150)
          newEventTime += 150;
        else if (fcurrentMicroS + newEventTime > 999750)
          newEventTime -= 150;

        if (fMCConfigFile.empty())
          EventGeneration(random, fcurrentMicroS + newEventTime, 
                          nStationsPerEvent);
        else 
          EventGeneration(random, fcurrentMicroS + newEventTime, 
                          eventTypeDistribution);
        newEventTime = eventDist(random);
      }

      if (fcurrentMicroS < 1000000) {
        t2.fTime = fcurrentMicroS;
        GetSingleT2Id(t2, random);
        //set trigger type roughly 20 : 1 (Th : ToT)
        if (dist(random) <= 5)
          t2.fTriggers = 9;
        else 
          t2.fTriggers = 1;
        if (abs(fLastTriggerTime[t2.fId] - t2.fTime) < 19)
          continue;
        fLastTriggerTime[t2.fId] = t2.fTime;
        foutput[fcurrentIndex++] = t2;
      }      
    }
    fcurrentMicroS -= 1000000;
    for (auto& t : fLastTriggerTime)
      t -= 1000000;

    outStream.write((char*)&fcurrentIndex, sizeof(int));
    outStream.write((char*)&foutput, sizeof(T2)*fcurrentIndex);
    ++fGPSSecond;
  }

  std::cout << "n(Events): " << fEventCounter << std::endl;
  outStream.close();
}


void 
MCGenerator::AddEvents(const std::string& inputFile, 
                       const std::string& outputfile,
                       const bool shuffle = false,
                       const int nStationsPerEvent = 2)
{
  std::mt19937_64 random(fSeed);
  if (!fSeed)
    random.seed(std::random_device()());
  std::exponential_distribution<double> eventDist(fEventRate/1e6);

  t2::T2DumpFile file(inputFile, fStationMask, shuffle, eventDist(random));
  std::ofstream outStream(outputfile, std::ios::binary);
  std::vector<double> weights;
  
  ReadConfigFile(weights);
  std::discrete_distribution<> eventTypeDistribution(weights.begin(), weights.end());

  double newEventTime = eventDist(random);
  
  t2::TwoSeconds ts;
  const std::vector<t2::T2Data>& t2s = ts.fT2s;
  const t2::T2Data oneSec(300e6/*m*/);

  if (!fEventRate)
    newEventTime = 3601*1e6;

  for (uint sec = 1; file.MergeNextSecond(ts) || t2s.size(); ++sec) {
    if (sec == 1)
      continue;
    file.ResetNRejected();

    fGPSSecond = ts.fGPSSecond;
    const unsigned int n = std::distance(t2s.begin(), 
                               lower_bound(t2s.begin(), t2s.end(), oneSec));
    
    outStream.write((char*)&fGPSSecond, sizeof(fGPSSecond));
    fcurrentIndex = 0;

    if (!(fGPSSecond % 300))
      std::cout << fGPSSecond << " nEvents: " << fEventCounter << std::endl;

    for (unsigned int i1 = 0; i1 < n; ++i1) {
      const t2::T2Data& t1 = t2s[i1];
      T2 t2(t1);  //interface to old format... also converts m to us
      foutput[fcurrentIndex++] = t2;
      const double timeDifference = 
              i1 ? t2.fTime - t2s[i1 - 1].fTime/t2::kMicroSecond : t2.fTime;
      
      newEventTime -= timeDifference;
      if (newEventTime <= 0) {
        ++fEventCounter;
        if (t2.fTime + newEventTime < 150)
          newEventTime += 150;
        else if (t2.fTime + newEventTime > 999750)
          newEventTime -= 150;

        if (fMCConfigFile.empty())
          EventGeneration(random, 
                          t2.fTime + newEventTime, 
                          nStationsPerEvent);
        else 
          EventGeneration(random, 
                          t2.fTime + newEventTime, 
                          eventTypeDistribution);
        newEventTime = eventDist(random);
      }
    }
    //end of second
    newEventTime -= 1e6;
    if (t2s.size() == n)//end of file    
      std::cout << "ending" << std::endl;
    outStream.write((char*)&fcurrentIndex, sizeof(int));
    outStream.write((char*)&foutput, sizeof(T2)*fcurrentIndex);
  }

  std::cout << "n(Events): " << fEventCounter << std::endl;
  outStream.close();
}


//output creates the vector of weights for a
// std::discrete_distribution out of the file
void
MCGenerator::ReadConfigFile(std::vector<double>& output)
{
  std::ifstream inConfig(fMCConfigFile);
  output.clear();

  double w = 0;
  int key = 0;  //should be type*100 + nStations

  int maxKey = 0;

  std::map<int, double> mapWeights;
  while (inConfig >> key >> w) {
    if (key < 0 || key > 499)
      throw std::runtime_error("invalid key!");

    if (key > maxKey)
      maxKey = key;
    auto it = mapWeights.find(key);
    if (it != mapWeights.end()) {
      std::cerr << "warning: event type " << key 
                << " appearing more than once in config file"
                << std::endl;
      it->second += w;
    } else {
      mapWeights.insert(std::make_pair(key, w));
    }
  }

  const auto itEnd = mapWeights.end();

  for (int i = 0; i <= maxKey; ++i) {
    const auto it = mapWeights.find(i);
    if (it != itEnd) 
      output.push_back(it->second);
    else
      output.push_back(0);
  }
}