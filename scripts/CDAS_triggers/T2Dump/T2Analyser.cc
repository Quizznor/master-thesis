#include <T2Dump/T2Analyser.h>
#include <fstream>
#include <cstdlib>
#include <time.h>
#include <iostream>
#include <iomanip>
#include <ostream>
#include <TMinuit.h>
#include <TH1D.h>
#include <TF2.h>
#include <Math/ProbFunc.h>
#include <TFile.h>
#include <TGraph.h>
#include <TString.h>
#include <utl/Accumulator.h>

#include <io/zfstream.h>
#include <utl/LineStringTo.h>
#include <t2/T2DumpFile.h>
#include <t2/StationInfo.h>
#include <t2/TwoSeconds.h>


typedef unsigned int uint;

/*T2Analyser::T2Analyser(std::string outbase) : 
  fOutReconstructed(outbase + "_Reconstructed.root"),
  fOutCandidates(outbase + "_Candidates.root"),
  fTSHist("TS","TS", 1000, 0, 1000),
  fTSvsnMaxHist("TSvsNMax","TSvsNMax", 1000, 0, 1000, 300, 0, 300),
  fForbushHist("forbushHist","forbushHist", 1600, -1, 3, 1600, -1, 3)
{
  srand(time(NULL));
}*/


T2Analyser::T2Analyser(const std::string& outbase,
                       const std::vector<std::string>& filenames, 
                       bool backgroundRandom) : 
  fDataHandler(outbase, filenames, backgroundRandom),
  fItCurrentT2(fDataHandler.begin()),
  fOutReconstructed(outbase + "_Reconstructed.root"),
  fOutCandidates(outbase + "_Candidates.root"),
  fTSHist("TS","TS", 1000, 0, 1000),
  fTSvsnMaxHist("TSvsNMax","TSvsNMax", 1000, 0, 1000, 300, 0, 300),
  fForbushHist("forbushHist","forbushHist", 1600, -1, 3, 1600, -1, 3),
  fFilenames(filenames)
{
  srand(time(NULL));
  ReadStationInfo(2000);
}


T2Analyser::~T2Analyser() 
{
  fOutReconstructed.Close();
  if (fForbushSearch)
    fOutCandidates.Write(fForbushHist);
  
  fOutCandidates.Write(fTSHist);
  //fOutCandidates.Write(fTSvsnMaxHist);
  fOutCandidates.Close();

}


void
T2Analyser::ReadStationInfo(const unsigned int maxNStations)
{
  fStationInfo.clear();
  fStationInfo.resize(maxNStations + 1);
  t2::StationInfo<int> meanStation;
  fStationMask.clear();
  fStationMask.resize(maxNStations + 1);
  int nStations = 0;

  t2::Vector<datatype> meanPos;

  io::zifstream ifs("t2/in-grid1_stations.dat.bz2");
  typedef std::istream_iterator<utl::LineStringTo<t2::StationInfo<datatype>>> Iterator;
  const std::vector<t2::StationInfo<datatype>> stations{Iterator(ifs), Iterator()};
  for (const auto& s : stations) {
    unsigned short int id = s.fId;
    if (id && id <= maxNStations) {
      fStationInfo[id] = s;
      fStationMask[id] = 1;
      meanPos += s.fPosition;
      ++nStations;
    }
  }
  meanPos /= nStations;
  for (auto& s : fStationInfo) {
    if (s)
      s.fPosition -= meanPos;
  }
}


void
T2Analyser::ReadPositions(std::string filename)
{
  std::ifstream inPositions(filename);
  double x = 0;
  double y = 0;
  double z = 0;
  int id = 0;

  double tmp = 0;
   
  //read in converted to micro seconds
  while (inPositions >> id >> y >> x >> z >> tmp >> tmp >> tmp >> tmp >> tmp >> tmp) {
    if (z) {
      fPositions[id].SetXYZ((x - 474000.92)/300.,
                         (y - 6100000.)/300. ,
                         (z - 1533.49)/300. );
    } else {
      fPositions[id].SetXYZ((x - 474000.92)/300.,
                            (y - 6100000.)/300. ,
                            (tmp - 1533.49)/300. );
    }
  }
}


void
T2Analyser::GetCompatibleT2s(
  std::vector<T2>& compatibleData, 
  int tolerance = 2)
{
  const double maxTimeDifference = GetMaximalTimeDifference(fItCurrentT2->fId);

  for (auto it = fItCurrentT2 + 1; it != fDataHandler.end(); ++it) {
    if (it->fId == fItCurrentT2->fId)
      continue;

    if (fPositions[it->fId].z() < -3. || !fPositions[it->fId].z())
      continue;
    
    const int deltaT = it->fTime - fItCurrentT2->fTime;

    if (deltaT > maxTimeDifference)
      break;

    const double deltaX2 = (fPositions[fItCurrentT2->fId] - fPositions[it->fId]).Mag2() 
                            - sqr((fPositions[fItCurrentT2->fId] - fPositions[it->fId]).z());

    if (sqr(deltaT - tolerance) <= deltaX2) {
      T2 tmp(*it);
      tmp.fTime = deltaT;
      compatibleData.push_back(tmp);
    }
  }
}


void
T2Analyser::FindTriplets(
  const std::vector<T2>& compatibleData,
  std::vector<T2Triplet<>>& triplets,
  double tolerance = 0.)
{
  T2Triplet<> tmpT2(fItCurrentT2->fTime);
  tmpT2.fIds[0] = fItCurrentT2->fId;

  for (auto it = compatibleData.begin(); it != compatibleData.end(); ++it) {
    tmpT2.fTimes[0] = it->fTime;
    tmpT2.fDeltaX[0] = (fPositions[it->fId] - fPositions[fItCurrentT2->fId]);
    tmpT2.fIds[1] = it->fId;

    for (auto it2 = it+1; it2 != compatibleData.end(); ++it2) {
      if (it2->fId == it->fId)
        continue;

      const int deltaT23 = it2->fTime - it->fTime;
      if ((fPositions[it->fId] - fPositions[it2->fId]).Mag2() <= sqr(deltaT23))
        continue;
      
      tmpT2.fTimes[1] = it2->fTime;
      tmpT2.fDeltaX[1] = (fPositions[it2->fId] - fPositions[fItCurrentT2->fId]);
      if (fabs(tmpT2.fDeltaX[1]*tmpT2.fDeltaX[0])/sqrt(tmpT2.fDeltaX[0].Mag2()*tmpT2.fDeltaX[1].Mag2()) > 0.99)
        continue;

      tmpT2.fIds[2] = it2->fId;
      if (!tmpT2.fTimes[0] && !tmpT2.fTimes[1]) {
        triplets.push_back(tmpT2);
        T2 tmp(*it);
        tmp.fTime += tmpT2.fMicroSecond;
        if (!std::count(fCandidate.fT2s.begin(), fCandidate.fT2s.end(), tmp)) {  
          fCandidate.fT2s.push_back(tmp);
        }
        T2 tmp2(*it2);
        tmp2.fTime += tmpT2.fMicroSecond;
        if (!std::count(fCandidate.fT2s.begin(), fCandidate.fT2s.end(), tmp2)) {
          fCandidate.fT2s.push_back(tmp2);
        }
      }

      if (tmpT2.IsInLightCone(tolerance)) {
        triplets.push_back(tmpT2);
        T2 tmp(*it);
        tmp.fTime += tmpT2.fMicroSecond;
        if (!std::count(fCandidate.fT2s.begin(), fCandidate.fT2s.end(), tmp)) {  
          fCandidate.fT2s.push_back(tmp);
        }
        T2 tmp2(*it2);
        tmp2.fTime += tmpT2.fMicroSecond;
        if (!std::count(fCandidate.fT2s.begin(), fCandidate.fT2s.end(), tmp2)) {
          fCandidate.fT2s.push_back(tmp2);
        }
      } 
    }
  }
}


int
T2Analyser::GetMaximalTimeDifference(ushort id)
  const 
{
  return std::min(double(150 + abs(fPositions[id].x()) 
                + abs(fPositions[id].y())), fMaxTimeDifference);
}


//find stations compatible with current reconstructed direction
// !in compatible data the microSecond of the first T2 is subtracted!
ushort 
T2Analyser::FindCompatibleToFit(ReconstructedT2& reconT2, 
      const T2Triplet<>& t, const std::vector<T2>& compatibleData)
{
  const TVector3 axis(reconT2.fu, reconT2.fv, sqrt(1 - sqr(reconT2.fu) - sqr(reconT2.fv)));
  const double inverseAvgDistance = 1./t.GetAvgDistance();

  for (const auto& t2 : compatibleData) {
    bool alreadyUsed = false;
    for (int i = 0; i < 3; ++i) {
      if (t2.fId == t.fIds[i])
        alreadyUsed = true;
    }
    if (alreadyUsed)
      continue;

    //c.f. e.g. TestTimeUncertainty.C (based on distances)
    double tolerance = 1.*(fPositions[t2.fId] - fPositions[fItCurrentT2->fId]).Mag()*inverseAvgDistance;
    if (tolerance < 1.)
      tolerance = 1.01;

    const double expectedTimeDifference = 
        -axis*(fPositions[t2.fId] - fPositions[fItCurrentT2->fId]);

    if (fabs(t2.fTime - expectedTimeDifference) < tolerance) {
      ++reconT2.fAdditionalMatches;
      /*std::cout << t2.fTime << " - " << expectedTimeDifference << " | "
                << fabs(t2.fTime - expectedTimeDifference) << " < " << tolerance << std::endl;*/
    }
  }
  return reconT2.fAdditionalMatches;
}


void
T2Analyser::IsCompatibleToFit(ReconstructedT2& r, const TVector3& axis,
    double avgDistance, const T2& t2)
  const
{
  const double tolerance = std::max(1.01, (fPositions[t2.fId] - fPositions[fItCurrentT2->fId]).Mag()/avgDistance);
  const double expectedTimeDifference = 
        -axis*(fPositions[t2.fId] - fPositions[fItCurrentT2->fId])/axis.Mag();

  if (fabs(t2.fTime - r.fMicroSecond - (expectedTimeDifference)) < tolerance)
    ++r.fAdditionalMatches;
}


//new interface
void
T2Analyser::CheckCompatibility(ReconstructedT2& r, const TVector3& axis,
                    double inverseAvgDistance, const t2::T2Data& t2) 
  const
{
  const int tolerance = std::max(301., 
            300.*sqrt(fStationInfo[t2.fId].Distance2(fStationInfo[r.fIds[0]]))*inverseAvgDistance);
  //std::cout << sqrt(fStationInfo[t2.fId].Distance2(fStationInfo[r.fIds[0]])) << std::endl;
  const t2::Vector<datatype> deltaPositions = fStationInfo[t2.fId].fPosition 
                                              - fStationInfo[r.fIds[0]].fPosition;

  const int expectedTimeDifference = - axis.x()*deltaPositions.fX 
                                     - axis.y()*deltaPositions.fY 
                                     - axis.z()*deltaPositions.fZ;
  const int timeDifferenceInMeter = abs(t2.fTime - r.fMicroSecond - int(expectedTimeDifference));

  if (timeDifferenceInMeter < tolerance) {
    ++r.fAdditionalMatches;

    //std::cout << t2.fTime << " - " << r.fMicroSecond << " - " << expectedTimeDifference << " | "
    //          << timeDifferenceInMeter << " < " << tolerance << std::endl;
  }
}

void
T2Analyser::FitTriplets(std::vector<T2Triplet<>>& triplets,
          const std::vector<T2>& compatibleData)
{
  for (auto& t : triplets) {
    double u = 0;
    double v = 0;
    bool fitSuccess = false;

    try {
      fitSuccess = t.GetStartValues(u, v);  
    } catch (std::exception& e) {
      std::cerr << "Exception: " << e.what() << std::endl;
      continue;
    }

    if (fitSuccess) {
      auto reco = ReconstructedT2(fDataHandler.fGPSSecond, t.fMicroSecond);
      
      reco.fu = u;
      reco.fv = v;

      reco.fDistance[0] = t.fDeltaX[0].Mag();
      reco.fDistance[1] = t.fDeltaX[1].Mag();
      reco.fDistance[2] = (t.fDeltaX[0] - t.fDeltaX[1]).Mag();
      
      for (int i = 0 ; i < 3; ++i)
        reco.fIds[i] = t.fIds[i];
      
      FindCompatibleToFit(reco, t, compatibleData);

      if (fForbushSearch) {
        if (reco.fDistance[0] < 4. || reco.fDistance[1] < 4. || reco.fDistance[2] < 4.)
          fForbushHist.Fill(reco.fu, reco.fv);
        else if (reco.fDistance[0] < 6. || reco.fDistance[1] < 6. || reco.fDistance[2] < 6.)
          fForbushHist.Fill(reco.fu + 2., reco.fv + 2);
      }

      if (fCandidate.IsCompatibleWithTimeRange(reco, fStepsize)) {
        fCandidate.AddReconstructedTriplet(reco);
      } else {
        CheckCandidate();
        fCandidate.AddReconstructedTriplet(reco);
      }

      //fReconstructed.push_back(reco);
      
      if (fFullOutput)
        fOutReconstructed << reco;
    }/* else {  //was used for using numerical fits for near horizon events -> too slow
      FitData::SetFitData(t);
      auto recoCand = FitHorizontalCandidate(u,v);
      if (!recoCand.fGPSSecond) //Migrad failed, returns default constructor
        continue;

      t.GetTestStat(recoCand);
      recoCand.fAverageDistance = t.GetAvgDistance();

      fOutHorizonReconstructed << recoCand;
    }*/
  }
}


//check if current candidate, a.k.a. 500 mus interval
// contains "interesting" signals
void
T2Analyser::CheckCandidate()
{
  fTSHist.Fill(fCandidate.fTestStatistic);
  
  if (fCandidate.fTestStatistic > fMinimalTestStatistic
      && fCandidate.fReconstructedTriplets.size() < 10000) {
    std::vector<std::pair<double, double> > directions;
    for (const auto& r : fCandidate.fReconstructedTriplets) {
      directions.emplace_back(r.fu, r.fv);
    }

    const auto labels = DBScanUV(directions, 0.005,
                              4, fCandidate.fnCluster, fCandidate.fnMaxPerCluster);

    for (const auto& l : labels) {
      if (l > 0)
        ++fCandidate.fnInCluster;
    }
  }
  fTSvsnMaxHist.Fill(fCandidate.fTestStatistic, fCandidate.fnMaxPerCluster);
  
  if (fCandidate.fTestStatistic > fMinimalTestStatistic 
      || fCandidate.fnMaxPerCluster > fMinNMaxPerCluster
      || fCandidate.fnInCluster > fnMinInCluster
      || fCandidate.fReconstructedTriplets.size() >= 10000
      || rand() < 100.) {
    fCandidate.Reconstruct();
    fCandidate.GetAvgDistance();

    fOutCandidates << fCandidate;

    std::cout << "fTestStatistic: " << std::setw(4) << fCandidate.fTestStatistic
            << " GPSs/microsecond: " << std::setw(12) << fDataHandler.fGPSSecond
            << " / " << std::setw(9) << fItCurrentT2->fTime << " " 
            << fCandidate.fSignalClass << std::endl;
  }

  fCandidate.Clear();  
}


void
T2Analyser::Analyse(double toleranceLightCone)
{
  if (!(fDataHandler.fGPSSecond % 10))
    std::cout << "Processing GPSsecond " << fDataHandler.fGPSSecond << std::endl;

  if (!fDataHandler.size()) {
    std::cerr << " warning input file was empty/not readable!" << std::endl;
    return;
  }

  bool errUnknownIds = false;

  std::vector<T2> compatibleData;
  std::vector<T2Triplet<>> triplets;

  while (!(fDataHandler.fEoF && 
        fDataHandler.IsSecond(fItCurrentT2 - fDataHandler.begin()))) {
    if (!fPositions[fItCurrentT2->fId].z() ||
      fPositions[fItCurrentT2->fId].Mag2() > 50000.) {
      errUnknownIds = true;
      ++fItCurrentT2;
      continue;
    }
    if (!std::count(fCandidate.fT2s.begin(), fCandidate.fT2s.end(), *fItCurrentT2))
      fCandidate.fT2s.push_back(*fItCurrentT2);

    //check if new second of data has to be loaded
    if (fDataHandler.IsSecond(fItCurrentT2 - fDataHandler.begin())) {
      int newIndexOfItCurrentT2 = fDataHandler.ReadNextSecond(fItCurrentT2 - fDataHandler.begin());
      fItCurrentT2 = fDataHandler.begin() + newIndexOfItCurrentT2;
      if (!(fDataHandler.fGPSSecond % 10))
        std::cout << "Processing GPSsecond " << fDataHandler.fGPSSecond << std::endl;
    }

    //actual reconstruction of triplets, iterating through T2s
    GetCompatibleT2s(compatibleData, 0.5);
    FindTriplets(compatibleData, triplets, toleranceLightCone);
    FitTriplets(triplets, compatibleData);

    compatibleData.clear();
    triplets.clear();

    ++fItCurrentT2;
  }

  if (errUnknownIds)
    std::cerr << "There were unknown Ids!" << std::endl;
}


void
T2Analyser::AnalyseReadOnly()
{
  if (!(fDataHandler.fGPSSecond % 10))
    std::cout << "Processing GPSsecond " << fDataHandler.fGPSSecond 
              << " data size: " << fDataHandler.size() 
              << std::endl;

  if (!fDataHandler.size()) {
    std::cerr << "warning input file was empty/not readable!" << std::endl;
    return;
  }

  bool errUnknownIds = false;

  while (!(fDataHandler.fEoF && 
          fDataHandler.IsSecond(fItCurrentT2 - fDataHandler.begin()))) {
    if (!fItCurrentT2->fId ||
        fPositions[fItCurrentT2->fId].Mag2() > 50000. || 
        !fPositions[fItCurrentT2->fId].z()) {
      errUnknownIds = true;
      ++fItCurrentT2;
      continue;
    }
    if (!std::count(fCandidate.fT2s.begin(), fCandidate.fT2s.end(), *fItCurrentT2))
      fCandidate.fT2s.push_back(*fItCurrentT2);
    

    //check if new second of data has to be loaded
    // and make sure that the iterator fItCurrentT2 remains valid
    if (fDataHandler.IsSecond(fItCurrentT2 - fDataHandler.begin())) {
      int newIndexOfItCurrentT2 = fDataHandler.ReadNextSecond(fItCurrentT2 - fDataHandler.begin());
      fItCurrentT2 = fDataHandler.begin() + newIndexOfItCurrentT2;
      
      if (!(fDataHandler.fGPSSecond % 10))
        std::cout << "Processing GPSsecond " 
                  << fDataHandler.fGPSSecond 
                  << " size: " << fDataHandler.size()
                  << std::endl;
    }
    //actual reconstruction of triplets, iterating through T2s
    const int maxTimeDifference1 = GetMaximalTimeDifference(fItCurrentT2->fId);
    T2Triplet<> triplet(*fItCurrentT2);

    for (auto itSecondT2 = fItCurrentT2 + 1;
         itSecondT2 != fDataHandler.end(); ++itSecondT2) { 
      if (!fPositions[itSecondT2->fId].z())
        continue;

      const int deltaT12 = itSecondT2->fTime - fItCurrentT2->fTime;
      if (deltaT12 > maxTimeDifference1)
        break;

      const bool addingSecondT2 = triplet.AddT2(*itSecondT2, 
                    fPositions[itSecondT2->fId] - fPositions[fItCurrentT2->fId]);
      if (!addingSecondT2) 
        continue;

      if (!std::count(fCandidate.fT2s.begin(), fCandidate.fT2s.end(), *itSecondT2))
        fCandidate.fT2s.push_back(*itSecondT2);

      const int maxTimeDifference2 = GetMaximalTimeDifference(itSecondT2->fId);
      
      for (auto itThirdT2 = itSecondT2 + 1;
           itThirdT2 != fDataHandler.end(); ++itThirdT2) {
        if (!fPositions[itThirdT2->fId].z())
          continue;
        const int deltaT23 = itThirdT2->fTime - itSecondT2->fTime;
        const int deltaT13 = itThirdT2->fTime - fItCurrentT2->fTime;
        if (deltaT13 > maxTimeDifference1 ||
            deltaT23 > maxTimeDifference2)
          break;

        const bool addThirdT2 = triplet.AddT2(*itThirdT2,
                      fPositions[itThirdT2->fId] - fPositions[fItCurrentT2->fId]);

        if (!addThirdT2)
          continue;

        if (!std::count(fCandidate.fT2s.begin(), fCandidate.fT2s.end(), *itThirdT2))
          fCandidate.fT2s.push_back(*itThirdT2);

        ReconstructedT2 reconstructedTriplet;
        try {
          reconstructedTriplet = triplet.ReconstructTriplet(fDataHandler.fGPSSecond);
        } catch(std::exception& e) {
          std::cerr << e.what() << std::endl;
          continue;
        }

        if (!reconstructedTriplet.fGPSSecond) {
          std::cerr << "invalid reconstruction" << std::endl;
          continue;
        }

        //checking for additional matching t2s
        const TVector3 axis(reconstructedTriplet.fu, reconstructedTriplet.fv, 
                            sqrt(1 - reconstructedTriplet.fu*reconstructedTriplet.fu 
                                 - reconstructedTriplet.fv*reconstructedTriplet.fv));
        const double avgDistance = triplet.GetAvgDistance();

        for (auto itAdditionalMatches = fItCurrentT2 + 1;
             itAdditionalMatches != fDataHandler.end(); ++itAdditionalMatches) {
          if (reconstructedTriplet.ContainsId(itAdditionalMatches->fId))
            continue;
          const int deltaT = itAdditionalMatches->fTime - fItCurrentT2->fTime;
          if (deltaT > maxTimeDifference1)
            break;

          IsCompatibleToFit(reconstructedTriplet, axis, 
                            avgDistance, *itAdditionalMatches);
        } 

        if (fFullOutput)
          fOutReconstructed << reconstructedTriplet;

        if (fCandidate.IsCompatibleWithTimeRange(reconstructedTriplet, fStepsize)) {
          fCandidate.AddReconstructedTriplet(reconstructedTriplet);
        } else {
          CheckCandidate();
          fCandidate.AddReconstructedTriplet(reconstructedTriplet);
        }
        //clear last t2
        triplet.RemoveT2(3);
      }
      //clear second t2
      triplet.RemoveT2(2);
    }

    ++fItCurrentT2;
  }

  if (errUnknownIds)
    std::cerr << "There were unknown Ids!" << std::endl;
}


void
T2Analyser::AnalyseNewInterface()
{
  typedef t2::StationInfo<datatype> SI;
  t2::TwoSeconds ts;
  const std::vector<t2::T2Data>& t2s = ts.fT2s;
  const t2::T2Data oneSec(300e6/*usec*/);

  const double maxR = 38200;//m
  const double maxDt = maxR;//us

  for (const auto& name : fFilenames) {
    std::cout << "using filename: " << name << std::endl;

    t2::T2DumpFile file(name, fStationMask);
    for (unsigned int sec = 1; file.MergeNextSecond(ts); ++sec) {
      if (sec == 1)
        continue;
      if (!(sec % 10))
        std::cout << sec << ' ' << ts.fGPSSecond << ' ' << t2s.size() << std::endl;
      const unsigned int m = t2s.size();
      const unsigned int n = std::distance(t2s.begin(), lower_bound(t2s.begin(), t2s.end(), oneSec));
      
      for (unsigned int i1 = 0; i1 < n; ++i1) {
        const t2::T2Data& t1 = t2s[i1];
        const SI& s1 = fStationInfo[t1.fId];
        const t2::T2Data maxT2(t1.fTime + maxDt + sqrt(s1.fPosition.XYMag2()));
        T2Triplet<t2::Vector<datatype>> triplet(t1);

        for (unsigned int i2 = i1 + 1; i2 < m && t2s[i2] < maxT2; ++i2) {
          const t2::T2Data& t2 = t2s[i2];
          const SI& s2 = fStationInfo[t2.fId];
          const t2::Vector<datatype> deltaPos21 = s2.fPosition - s1.fPosition;
    
          const bool addSecondT2 = triplet.AddT2(t2, deltaPos21);
          if (!addSecondT2) 
            continue;

          const t2::T2Data maxSecondT2(t2.fTime + maxDt + sqrt(s1.fPosition.XYMag2()));
          
          for (unsigned int i3 = i2 + 1;
               i3 < m && t2s[i3] < maxT2 && t2s[i3] < maxSecondT2; ++i3) {
            const t2::T2Data& t3 = t2s[i3];
            const SI& s3 = fStationInfo[t3.fId];
            const t2::Vector<datatype> deltaPos31 = s3.fPosition - s1.fPosition;
            
            const bool addThirdT2 = triplet.AddT2(t3, deltaPos31);
            if (!addThirdT2) 
              continue;

            ReconstructedT2 reconstructedTriplet;
            try {
              reconstructedTriplet = triplet.ReconstructTriplet(ts.fGPSSecond);
            } catch(std::exception& e) {
              std::cerr << e.what() << std::endl;
              continue;
            }

            if (!reconstructedTriplet.fGPSSecond) {
              std::cerr << "invalid reconstruction" << std::endl;
              continue;
            }
            
            //checking for additional matching t2s
            //add factor 100 to use integer arithmetic of t2::Vector
            //std::cout << "fu: " << reconstructedTriplet.fu 
            //          << " fv: " << reconstructedTriplet.fv << std::endl;
            const TVector3 axis(reconstructedTriplet.fu, reconstructedTriplet.fv, 
                                sqrt(1 - reconstructedTriplet.fu*reconstructedTriplet.fu 
                                     - reconstructedTriplet.fv*reconstructedTriplet.fv));
            const double inverseAvgDistance = 1./triplet.GetAvgDistance();

            for (unsigned int iAdd = i1 + 1; iAdd < m && t2s[iAdd] < maxT2; ++iAdd) {
              const t2::T2Data& tAdd = t2s[iAdd];
              if (reconstructedTriplet.ContainsId(tAdd.fId))
                continue;
              
              CheckCompatibility(reconstructedTriplet, axis, 
                                 inverseAvgDistance, tAdd);
            }

            if (fFullOutput)
              fOutReconstructed << reconstructedTriplet;

            if (fCandidate.IsCompatibleWithTimeRange(reconstructedTriplet, fStepsize, 1./300.)) {
              fCandidate.AddReconstructedTripletMeterBased(reconstructedTriplet);
            } else {
              CheckCandidate();
              fCandidate.AddReconstructedTripletMeterBased(reconstructedTriplet);
            }
          }
          triplet.RemoveT2(3);
        }
        triplet.RemoveT2(2);
      }
    }
  }
}

/*void
T2Analyser::FillFitData(const T2Triplet& triplet)
{
  for (int i = 0; i < 3; ++i)
    FitData::fTimes[i] = triplet.fTimes[i];

  FitData::fdeltaX[1] = triplet.fDeltaX[1];
  FitData::fdeltaX[2] = triplet.fDeltaX[2];
}


FittedT2
T2Analyser::FitHorizontalCandidate(double thetaStart, double phiStart)
{
  TMinuit m(3);
  m.SetFCN(FitData::fcnGausThetaPhi);
  m.SetPrintLevel(-1);
  m.DefineParameter(0, "t0", FitData::fTimes[0],
           0.5, -1000 + FitData::fTimes[0],
           FitData::fTimes[0] + 1000);
  m.DefineParameter(1, "theta", thetaStart, 0.2, -5, 5);
  m.DefineParameter(2, "phi", phiStart, 0.5, -10, 10);

  int errFlag = m.Migrad();
  if (errFlag) 
    return FittedT2();

  double fittedValues[3];
  double fitErrors[3];

  for (int i = 0; i < 3 ; ++i) 
    m.GetParameter(i, fittedValues[i], fitErrors[i]);

  FittedT2 r(fDataHandler->fGPSSecond);
  r.fT0 = fittedValues[0];
  r.fTheta = fittedValues[1];
  r.fPhi = fittedValues[2];
  
  r.fT0Err = fitErrors[0];
  r.fThetaErr = fitErrors[1];
  r.fPhiErr = fitErrors[2];

  double tmp;
  int npar = 3;
  FitData::fcnGausThetaPhi(npar, &tmp, r.fFcn, fittedValues, 0);

  return r;
}


void
T2Analyser::DumpIterator(std::ostream& o, DataIterator& it)
{
  o << it->fTime << " " << fPositions[it->fId].X() << " "
    << fPositions[it->fId].Y() << " "
    << fPositions[it->fId].Z() << std::endl;
}*/
