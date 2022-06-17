//#include <T2Dump/DataIterator.h>
#include <T2Dump/DataHandler.h>
#include <T2Dump/Utl.h>
#include <TVector3.h>
#include <string>
#include <io/RootOutFile.h>
#include <interface/RecoT2.h>
#include <interface/Events.h>
#include <interface/FittedT2.h>
#include <T2Dump/FitData.h>
#include <T2Dump/T2Triplet.h>
#include <t2/StationInfo.h>
#include <t2/T2Data.h>
#include <TH2D.h>
#include <TH1D.h>
#include <TF1.h>
#include <map>

typedef long datatype;

class T2Analyser
{
private:
  //'old' way of using data (potentially not needed later ... )
  DataHandler fDataHandler;
  Positions fPositions;
  std::vector<T2>::const_iterator fItCurrentT2;    //int as sign indicates errors in reading

  T2EventCandidate fCandidate;

  //output
  io::RootOutFile<ReconstructedT2> fOutReconstructed;
  io::RootOutFile<T2EventCandidate> fOutCandidates;

  TH1D fTSHist;
  TH2D fTSvsnMaxHist;
  TH2D fForbushHist;

  //new input format/style
  std::vector<std::string> fFilenames;  

  std::vector<t2::StationInfo<datatype>> fStationInfo;
  std::vector<char> fStationMask;


  double fMinimalTestStatistic = 50.;
  int fnMinInCluster = 25.;
  int fMinNMaxPerCluster = 30;

  //init functions:
  void
  ReadStationInfo(const unsigned int maxNStations);

  //triplet-reco based:
  void GetCompatibleT2s(std::vector<T2>& compatibleData, 
                        int tolerance);
  void FindTriplets(const std::vector<T2>& compatibleData, 
                    std::vector<T2Triplet<>>& triplets,
                    double tolerance);

  void FitTriplets(std::vector<T2Triplet<>>& triplets,
      const std::vector<T2>& compatibleData);
  void FillFitData(const T2Triplet<>& triplet);

  ushort FindCompatibleToFit(ReconstructedT2& reconT2,
      const T2Triplet<>& t, const std::vector<T2>& compData);


  void
  IsCompatibleToFit(ReconstructedT2& r, const TVector3& axis,
                    double avgDistance, const T2& t2) const;

  void
  CheckCompatibility(ReconstructedT2& r, const TVector3& axis,
                    double avgDistance, const t2::T2Data& t2) const;

  FittedT2 FitHorizontalCandidate(double uStart, double vStart);

  int GetMaximalTimeDifference(ushort id) const;

  //Check if the current candidate (fCandidate) is an event
  // if yes: write to file
  //resets the candidate as well
  void CheckCandidate();

public:
  T2Analyser(const std::string& outbase, 
             const std::vector<std::string>& filenames, 
             bool backgroundRandom);
  //T2Analyser(std::string outbase);
  ~T2Analyser();

  //Parameters for cuts in algorithms
  double fMaxTimeDifference = 250.; //maximal timedifference (cut before other checks)
  int fStepsize = 500;
  bool fForbushSearch = false;      //produce a histogram of 2 station directions and write it to file
  int fDumpSecond = 1149964100;
  bool fFullOutput = false;
  void SetMinimalTestStatisticValue(double ts) { fMinimalTestStatistic = ts; }
  double GetMinimalTestStatisticValue() const { return fMinimalTestStatistic; }

  //void SetDataHandler(DataHandler* h) { fDataHandler = h; fCurrentT2 = h->begin(); } 

  void ReadPositions(std::string filename = "/home/schimassek/SVN/T2Scalers/ms/src/Data/SdPositions.txt");
  void Analyse(double toleranceLightCone = 0.);
  void AnalyseReadOnly();
  void AnalyseNewInterface();

  void DumpIterator(std::ostream&, DataIterator& it);
};
