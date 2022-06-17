#ifndef _MCGen_
#define _MCGen_

#include <random>
#include <TVector3.h>
#include <string>
#include <interface/T2.h>
#include <interface/RecoT2.h>
#include <interface/Events.h>
#include <io/RootOutFile.h>
#include <T2Dump/Utl.h>
#include <T2Dump/Units.h>
#include <map>

typedef unsigned short ushort;

class MCGenerator
{
  //Positions fPositions;
  //const static std::map<ushort, ushort> fGridType;
  std::vector<t2::StationInfo<double>> fStationInfo;
  std::vector<char> fStationMask;
  std::string fMCConfigFile = "";   //file with spectrum for event types

  std::vector<double> fLastTriggerTime;
  T2 foutput[100000];
  int fcurrentIndex = 0;
  double fcurrentMicroS = 0;
  std::discrete_distribution<> fEventTriggerType;  // 2:1 ToT:Th-T2

  double fCosTheta = -1;    //for fixed theta angles. < 0 means random
  io::RootOutFile<MCEvent> fOutput;
  int fEventCounter = 0;

  void ReadConfigFile(std::vector<double>&);

  void GetSingleT2Id(T2& t2, std::mt19937_64& rand);

  void EventGeneration(std::mt19937_64& rand, int microSecond,
                       std::discrete_distribution<>& p);
  void EventGeneration(std::mt19937_64& rand, int microSecond,
                       uint nStations);

  void CreateEvent(std::mt19937_64& rand, const TVector3& direction, int microSecond);
  void CreateEvent(std::mt19937_64& rand, const TVector3& direction, int microSecond, uint nStations);
  //use neighbouring stations for a 'real' shower toy MC
  void CreateShower(std::mt19937_64& rand, const TVector3& direction, int microSecond);
  void CreateShower(std::mt19937_64& rand, const TVector3& direction, int microSecond, uint nStations);

  void CreateSpallationEvent(std::mt19937_64& rand, const TVector3& direction, int microSecond);
  void CreateSpallationEvent(std::mt19937_64& rand, const TVector3& direction, int microSecond, uint nStations);

  void CreateRingEvent(std::mt19937_64& rand, int microSecond);
  void CreateRingEvent(std::mt19937_64& rand, int microSecond, uint nStations);

  void CreateHorizontalShower(std::mt19937_64& rand, const TVector3& direction, int microSecond);
  void CreateHorizontalShower(std::mt19937_64& rand, const TVector3& direction, int microSecond, uint nStations);

  double GetMinimalDistance(const std::vector<ushort>& ids, ushort id) const;

public:
  MCGenerator() :
    fEventTriggerType({0, 1, 0, 0, 0, 0, 0, 0, 2}),
    fOutput("Test.root")
  {
    fLastTriggerTime.resize(2000, 0);
    ReadStationInfo(2000, fStationInfo, fStationMask);
    for (uint i = 0; i < 2000; ++i) {
      if (!fStationMask[i])
        continue;
      const auto& pos1 = fStationInfo[i].fPosition;
      for (uint j = i + 1; j < 2000; ++j) {
        if (!fStationMask[j])
          continue;
        const auto& pos2 = fStationInfo[j].fPosition;

        if (std::sqrt(pos1.Distance2(pos2)) < 100)
          fStationMask[j] = 0;
      }
    }
  }

  ~MCGenerator() { fOutput.Close(); }

  MCGenerator(const std::string& EventOutputFileName,
              double totweight = 2) :
    fEventTriggerType({0, 1, 0, 0, 0, 0, 0, 0, totweight}),
    fOutput(EventOutputFileName)
  {
    fLastTriggerTime.resize(2000, 0);
    ReadStationInfo(2000, fStationInfo, fStationMask);
    for (uint i = 0; i < 2000; ++i) {
      if (!fStationMask[i])
        continue;
      const auto& pos1 = fStationInfo[i].fPosition;
      for (uint j = i + 1; j < 2000; ++j) {
        if (!fStationMask[j])
          continue;
        const auto& pos2 = fStationInfo[j].fPosition;

        if (std::sqrt(pos1.Distance2(pos2)) < 100)
          fStationMask[j] = 0;
      }
    }
  }

  uint fGPSSecond = 1;
  bool fRealShower = false;
  int fMaximalNumberOfStationsPerEvent = 20;
  int fSeed = 13;
  short fType = -1;  //-1: chose at random. 0: compact, 1: extended
                     // 2: spallation 3: lightning

  double fEventRate = 0.33;

  void SetCosTheta(double cth);
  double GetCosTheta() { return fCosTheta; }
  void GenerateT2(int n, const std::string& outfilename,
                  int nPerSec, int nStationsPerEvent);

  void AddEvents(const std::string& inputFile,
                 const std::string& outputfile,
                 const bool shuffle,
                 const int nStationsPerEvent);

  void SetConfigFile(const std::string& file) { fMCConfigFile = file; }
  //void ReadPositions(const std::string& filename = "/home/schimassek/SVN/T2Scalers/ms/src/Data/SdPositions.txt");
};

#endif