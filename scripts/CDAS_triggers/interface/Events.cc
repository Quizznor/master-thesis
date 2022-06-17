#include <interface/Events.h>
#include <T2Dump/Utl.h>
#include <map>
#include <exception>

int 
T2EventCandidate::Classify()
{
  //TODO!
  return 0;
}

bool
T2EventCandidate::IsCompatibleWithTimeRange(const ReconstructedT2& r, 
  int windowsize, double conversionFactor) 
  const
{
  if (!fReconstructedTriplets.size())
    return true;

  const double timeDifferenceInMuS 
    = (r.fGPSSecond - fReconstructedTriplets.front().fGPSSecond)*1e6*conversionFactor
      + r.fMicroSecond*conversionFactor 
      - fReconstructedTriplets.front().fMicroSecond;

  if (fabs(timeDifferenceInMuS) < windowsize) {
    return true;
  } else {
    if (fTestStatistic > 20 && r.fAdditionalMatches > 1) 
      return true;
    else
      return false;
  }
}

void
T2EventCandidate::AddReconstructedTriplet(const ReconstructedT2& recoT2)
{
  fReconstructedTriplets.push_back(recoT2);

  if (recoT2.fAdditionalMatches > fMaxAdditionalMatches)
    fMaxAdditionalMatches = recoT2.fAdditionalMatches;

  if (recoT2.fAdditionalMatches > 2)  //was 2
    fTestStatistic += (recoT2.fAdditionalMatches - 2);
}


//adjust to cope with changed units in new file reading
void
T2EventCandidate::AddReconstructedTripletMeterBased
  (const ReconstructedT2& recoT2)
{
  ReconstructedT2 r = recoT2;
  r.fMicroSecond /= 300.;
  for (int i = 0; i < 3; ++i)
    r.fDistance[i] /= 300.;

  fReconstructedTriplets.push_back(r);

  if (r.fAdditionalMatches > fMaxAdditionalMatches)
    fMaxAdditionalMatches = r.fAdditionalMatches;

  if (r.fAdditionalMatches > 2)  //was 2
    fTestStatistic += (r.fAdditionalMatches - 2);
}

//calculate the microsecond and directions by using mean-values
void
T2EventCandidate::Reconstruct()
{
  utl::Accumulator::Mean mU;
  utl::Accumulator::Mean mV;
  utl::Accumulator::Mean mMicroSecond;

  const uint firstGPSs = fReconstructedTriplets.front().fGPSSecond;

  if (fTestStatistic > 2) {
    for (const auto& r : fReconstructedTriplets) {
      if (r.fAdditionalMatches > 1) {
        if (r.fGPSSecond == firstGPSs)
          mMicroSecond(r.fMicroSecond);
        else
          mMicroSecond(r.fMicroSecond + 1e6);

        mU(r.fu);
        mV(r.fv);
      }
    }
  } else {
    for (const auto& r : fReconstructedTriplets) {
      mU(r.fu);
      mV(r.fv);
    }
  }

  fu = mU.GetMean();
  fv = mV.GetMean();

  fMicroSecond = mMicroSecond.GetMean();
  if (fMicroSecond < 1e6) {
    fGPSSecond = firstGPSs;
  } else {
    fGPSSecond = firstGPSs + 1;
    fMicroSecond -= 1e6;
  }
  Classify();
}

double
T2EventCandidate::GetAvgDistance()
{
  int n = 0;
  if (fTestStatistic > 20.) {
    for (const auto& r : fReconstructedTriplets) {
      if (r.fAdditionalMatches > 2) {
        fAvgDistance += (r.fDistance[0] + r.fDistance[1] + r.fDistance[2]);
        ++n;
      }
    }  
  } else {
    for (const auto& r : fReconstructedTriplets) {
      fAvgDistance += (r.fDistance[0] + r.fDistance[1] + r.fDistance[2]);
      ++n;
    }  
  }
  
  fAvgDistance /= 3*n;
  return fAvgDistance;
}

//retrieve number of distances < threshold in distances of stations in triplets
int 
T2EventCandidate::GetNCompact(double threshold) 
  const
{
  int n = 0;
  for (const auto& r : fReconstructedTriplets) {
    for (int i = 0; i < 3; ++i)
      if (r.fDistance[i] < threshold)
        ++n;
  }
  return n;
}

//resets the important parts, not everything to save computation time
// The reason being, that in the case of a new event (real)
// the reconstruction method will be called anyway, which will
// overwrite the old values.
void
T2EventCandidate::Clear()
{
  fReconstructedTriplets.clear();
  fT2s.clear();

  fMaxAdditionalMatches = 0;
  fTestStatistic = 0;
  fnMaxPerCluster = 0;
  fnCluster = 0;
  fnInCluster = 0;
  fSignalClass = 0;
  fAvgDistance = 0;
  fu = 0;
  fv = 0;

  fGPSSecond = 0;
  fMicroSecond = 0;
}

//implement T3/compact rejection on triplet base
bool
T2EventCandidate::IsT3()
  const
{
  //check triplets
  for (const auto& triplet : fReconstructedTriplets) {
    //à la equilateral 3ToT
    if (triplet.fDistance[0] < 6. 
      && triplet.fDistance[1] < 6. 
      && triplet.fDistance[2] < 6.)
      return true;

   //à la strechted 3ToT
    int nNeighbours = 0;
    int nStrechted = 0;

    for (int i = 0; i < 3; ++i) {
      if (triplet.fDistance[i] < 6.) {
        ++nNeighbours;
      } else if (triplet.fDistance[i] < 9.) {
        ++nStrechted;
      }
    }

    if (nNeighbours == 2 && nStrechted == 1)
      return true;
  }
  return false;
}

//implements check if the T2s contain a T3, not the triplets
bool
T2EventCandidate::ContainsT3(const Positions& pos)
  const
{
  //construct `real' T3s: 3ToT-2C_1&3C_2 and 4T2-2C1&3C2&4C4 modes
  for (auto it = fT2s.begin(); it != fT2s.end(); ++it) {
    int nNeighboursToT[2] = {0, 0};         //count stations in C1, C2
    int nNeighboursAny[4] = {0, 0, 0, 0};   // ... C1, C2, C3, C4
  
    for (auto it2 = fT2s.begin(); it2 != fT2s.end(); ++it2) {
      if (it2->fId == it->fId)
        continue;

      //convert to meters from mus (*300.)
      const int crownCDAS = crown(pos[it->fId].x()*300., pos[it2->fId].x()*300.,
                                  pos[it->fId].y()*300., pos[it2->fId].y()*300.);
      const int deltaT = it->fTime - it2->fTime;

      //check compactness and timing conditions
      // hexagons are approximated with circles -> overestimation of compactness
      // time condition is (3 + 5*n) mu s, in CDAS XbAlgo.cc (<= 5*neighbour + dtime)
      // with dtime = 3
      if (fabs(deltaT) > 3 + 5*crownCDAS)                 //directly remove out of time candidates
        continue; 
      if (crownCDAS == 1) {         //C1 neighbours
        if (it->IsToT() && it2->IsToT()) {
          ++nNeighboursAny[0];
          ++nNeighboursAny[1];

          for (int i = 0; i < 4; ++i)
            ++nNeighboursAny[i];
        } else {
          for (int i = 0; i < 4; ++i)
            ++nNeighboursAny[i];
        }
      } else if (crownCDAS == 2) {  //C2
        if (it->IsToT() && it2->IsToT()) {
          ++nNeighboursAny[1];

          for (int i = 1; i < 4; ++i)
            ++nNeighboursAny[i];
        }
      } else if (crownCDAS == 3) {  //C3
        for (int i = 2; i < 4; ++i)
            ++nNeighboursAny[i];
      } else if (crownCDAS == 4) {  //C4
          ++nNeighboursAny[3];
      }
    } 
    
    //3ToT trigger (using, that the center is triggered by construction)
    if (it->IsToT()
        && nNeighboursToT[0] >= 1 
        && nNeighboursToT[1] >= 2)
      return true;

    //4T2 mode 2C1&3C2&4C4
    if (nNeighboursAny[0] >= 1 
        && nNeighboursAny[1] >= 2 
        && nNeighboursAny[3] >= 3)
      return true;

    /*std::cout << "Id: " << it->fId 
              << " Position: " << pos[it->fId].x()*300. + 474000 << " , "
              << pos[it->fId].y()*300. + 6100000 
              << " nNeighboursAny[]: ";
    for (int i = 0; i < 4; ++i)
      std::cout << nNeighboursAny[i] << " ";
    std::cout << std::endl;*/
  }

  return false;
}

//search for straight lines in x-y projection of T2s
// indication for theta approx. 90 degree events, not found by T3s
TH2D
T2EventCandidate::HughTransformLineSearch(const Positions& pos)
  const
{
  TH2D outputhist("hough", "hough", 90, 0, 180, 200, 0, 50000);

  for (auto it = fT2s.begin(); it != fT2s.end(); ++it) {
    for (auto it2 = it + 1; it2 != fT2s.end(); ++it2) {
      const auto result = GetRTheta(pos[it->fId], pos[it2->fId], 450, 6070);
      outputhist.Fill(result.second*180./3.1415, result.first);
    }
  }

  return outputhist;
}

ushort
T2EventCandidate::GetNIdsWithMultipleT2s() 
  const
{
  std::map<ushort, ushort> ids;

  for (const auto& t2 : fT2s) {
    try{
      ++ids.at(t2.fId);
    } catch(std::exception& e) {
      ids.emplace(t2.fId, 1);
    }
  }

  ushort nMultiple = 0;
  for (const auto& n : ids) {
    if (n.second > 1)
      ++nMultiple;
  }

  return nMultiple;
}