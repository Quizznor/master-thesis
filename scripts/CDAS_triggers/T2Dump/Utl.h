#ifndef _utlInter_
#define _utlInter_

#include <vector>
#include <utility>
#include <iostream>
#include <map>

#include <TVector3.h>

#include <t2/Vector.h>
#include <t2/StationInfo.h>
#include <t2/T2Data.h>

#include <io/zfstream.h>

#include <utl/LineStringTo.h>
#include <utl/Accumulator.h>

#include <T2Dump/Units.h>


double
ModifiedJulianDate(uint gpsSecond);


time_t
GetUnixSecond(int year, int month, int day, int hour, int minute, int second);


uint
YMDtoGPSs(int year, int month, int day, int hour = 0, int minute = 0, int second = 0);


double
GPSsecond2GMST(uint gpsSecond);


double
CalculateNutationCorrection(const double julianDay);


std::vector<double>
ConvertToRaDec(const std::vector<double>& p);


std::vector<double>
GetGalacticCoordinates(const std::vector<double>& p);


//split x2 into a part parallel to x1, and a part perpendicular in 2D (ignoring z)
std::pair<TVector3, TVector3>
Get2DBasis(const TVector3& x1, const TVector3& x2);


//pairs: theta, phi
double
DirectionalDistance(const std::pair<double, double>& p1,
                    const std::pair<double, double>& p2);


std::vector<uint>
RangeQuery(const std::vector<std::pair<double, double> >& directions,
          double epsilon,
          const std::pair<double, double>& p);


//the number of points is excluding the current one (so in the range [0, inf))
std::vector<int>
DBScan(const std::vector<std::pair<double, double> >& directions,
      double epsilon,
      uint minPoints,
      int& nCluster,
      int& nMaxPerCluster);


//DBScan for u/v as the upper implementations is explicitly for angles
std::vector<uint>
RangeQueryUV(const std::vector<std::pair<double, double> >& directions,
             double epsilon,
             const std::pair<double, double>& p);


std::vector<int>
DBScanUV(const std::vector<std::pair<double, double> >& directions,
      double epsilon,
      uint minPoints,
      int& nCluster,
      int& nMaxPerCluster);


//intended to avoid unnecessary calls of Distance2()
// to save time, by first getting the upper limit of
// triplets by counting triplets in t0 window
// scans forward and backwards in the vector around given index
template<class T>
std::pair<int, uint>
CountCloseBy(const std::vector<T>& sortedInput,
             const unsigned index,
             double truncationLimit)
{
  const auto& startElement = sortedInput[index];
  uint cutIndexForward = index + 1;
  for (uint n = sortedInput.size(); cutIndexForward < n; ++cutIndexForward) {
    const auto& testElement = sortedInput[cutIndexForward];
    if (testElement.ft0 - startElement.ft0 > truncationLimit)
      break;
  }

  int cutIndexBackwards = int(index) - 1;
  for ( ; cutIndexBackwards >= 0; --cutIndexBackwards) {
    const auto& testElement = sortedInput[cutIndexBackwards];
    if (startElement.ft0 - testElement.ft0 > truncationLimit)
      break;
  }

  return std::make_pair(++cutIndexBackwards, cutIndexForward);
  //++ to get the first element and not the one before the first
}


//fills the indices of neighbours into output
// needs m to avoid unnecessary Distance2() calls
//  index: position of current point of DBScan (center)
//  truncationLimit: truncate the distance calculation to save time
//                   too big: very slow
//                   too small: potential loss of events
template<class T>
void
RangeQuery(const std::vector<T>& sortedInput,
           std::vector<unsigned>& output,
           const double epsilon,
           const unsigned m,
           const unsigned index,
           double truncationLimit)
{
  output.clear();
  const auto tmp = CountCloseBy(sortedInput, index, truncationLimit);
  if (int(tmp.second) - tmp.first < int(m))
    return;

  const T& point = sortedInput[index];
  for (uint i = tmp.first; i < tmp.second; ++i) {
    if (i == index)
      continue;
    const auto& testPoint = sortedInput[i];
    if (point.Distance2(testPoint) < epsilon)
      output.push_back(i);
  }
}


//implements a version of DBScan, assumes sorted input with respect to time
// this condition is not checked for performance reasons!
// in the case of unsorted input, the output might be wrong
// requires the class T to have
//   - a method double Distance2(const T&) const.
//   - a member 'fClusterLabel'
// class C is meant to be a cluster, needs
//   - a default constructor (emplace_back() is called)
//   - a method AddTriplet(const T& )
//   - a method type TimeDistance(const T& t) const;
// epsilon: 'search radius' of DBScan in terms of Distance2
// minPoints: minimal number of points inside a sphere of radius sqrt(epsilon)
template<class T, class C>
void
DBScan(std::vector<T>& sortedInput,
       const double epsilon,
       const unsigned minPoints,
       std::vector<C>& clusterOutput,
       C&,  //was noise but is not used, see commented code
       double truncationLimit = 50)
{
  //std::vector<short> labels;
  //labels: 0 undefined, > 0: cluster number; -1: noise
  // labels the directions in the directions vector according to clusters
  //labels.resize(sortedInput.size(), 0);
  short labelCounter = clusterOutput.size();
  std::vector<unsigned> neighbours;
  std::vector<unsigned> additionalNeighbours;

  if (!truncationLimit) {
    std::cerr << "warning: no truncation set,"
                 " falling back to no truncation!"
              << std::endl;
    truncationLimit = 1e6;
  }


  for (unsigned int i = 0; i < sortedInput.size(); ++i) {
    if (sortedInput[i].fClusterLabel)
      continue;

    RangeQuery(sortedInput, neighbours, epsilon, minPoints, i, truncationLimit);

    if (neighbours.size() < minPoints) {
      sortedInput[i].fClusterLabel = -1;
      continue;
    }
    sortedInput[i].fClusterLabel = ++labelCounter;
    clusterOutput.emplace_back();
    clusterOutput[labelCounter - 1].AddTriplet(sortedInput[i]);

    //while loop, as the vector might be altered by adding additional points
    // to the cluster; while to make clear, that this is not a simple for loop
    // neighbours are the indices of the triplets in directions that are
    // neighbours of the triplet-direction directions[i]
    uint index = 0;
    while (index < neighbours.size()) {
      if (sortedInput[neighbours[index]].fClusterLabel < 0) {
        sortedInput[neighbours[index]].fClusterLabel = labelCounter;
        clusterOutput[labelCounter - 1].AddTriplet(sortedInput[neighbours[index]]);
      } else if (sortedInput[neighbours[index]].fClusterLabel) {
        ++index;
        continue;
      }

      sortedInput[neighbours[index]].fClusterLabel = labelCounter;
      clusterOutput[labelCounter - 1].AddTriplet(sortedInput[neighbours[index]]);

      RangeQuery(sortedInput, additionalNeighbours,
                 epsilon, minPoints,
                 neighbours[index], truncationLimit);

      if (additionalNeighbours.size() < minPoints) {
        ++index;
        continue;
      }

      for (const auto& j : additionalNeighbours) {
        if (std::find(neighbours.begin(), neighbours.end(), j) == neighbours.end())
          neighbours.push_back(j);
      }

      ++index;
    }
  }

  //evaluating noise contribution for output
  if (!labelCounter)
    return;
  /*
  for (unsigned int i = 0; i < sortedInput.size(); ++i) {
    if (sortedInput[i].fClusterLabel >= 0)
      continue;

    for (const auto& c : clusterOutput) {
      if (abs(c.TimeDistance(sortedInput[i])) < 250.) {
        noise.AddTriplet(sortedInput[i]);
        break;
      }
    }
  }*/
}


std::pair<double, double>
GetRTheta(const TVector3& p1, const TVector3& p2, double x0 = 450, double y0 = 6070);


int crown(double x1, double x2, double y1, double y2); //copy from CDAS


template <typename T>
void
mergeSortedVectors(std::vector<T>& remaining,
                   const std::vector<T>& toInsert,
                   const T& initVal)
{
  std::vector<T> tmp(remaining.size() + toInsert.size(), initVal);
  auto it1 = remaining.begin();
  auto it2 = toInsert.begin();

  auto itInsert = tmp.begin();

  while (!(it1 == remaining.end() && it2 == remaining.end())) {
    if (it1 == remaining.end()) {
      tmp.insert(itInsert, it2, toInsert.end());
      break;
    }
    if (it2 == toInsert.end()) {
      tmp.insert(itInsert, it1, remaining.end());
      break;
    }

    if (*it1 < *it2) {
      *itInsert = *it1;
      ++itInsert;
      ++it1;
    } else {
      *itInsert = *it2;
      ++it2;
      ++itInsert;
    }
  }

  remaining = tmp;
}


//copy from /dv/src/interferometrix.cxx
template<typename T>
t2::Vector<T>
ReadStationInfo(const unsigned int maxNStations,
                std::vector<t2::StationInfo<T>>& stationInfos,
                std::vector<char>& stationMask,
                const std::string& stationFileName = "configs/idealArray.cfg",
                const std::vector<int>& allowedGridTypes = {},
                const unsigned meanGPSSecond = 0)
{
  stationInfos.clear();
  stationInfos.resize(maxNStations + 1);
  stationMask.clear();
  stationMask.resize(maxNStations + 1);
  t2::Vector<T> meanPos;
  int nStations = 0;

  io::zifstream ifs(stationFileName);
  typedef std::istream_iterator<utl::LineStringTo<t2::StationInfo<T>>> Iterator;
  const std::vector<t2::StationInfo<T>> stations{Iterator(ifs), Iterator()};
  for (const auto& s : stations) {
    unsigned short int id = s.fId;
    if (s && id <= maxNStations
        && s.IsValid(meanGPSSecond, allowedGridTypes)) {
      stationInfos[id] = s;
      stationMask[id] = 1;
      meanPos += s.fPosition;
      ++nStations;
    }
  }
  if (!nStations)
    throw std::runtime_error("ReadStationInfo() [Utl.h]: no valid stations found!");

  meanPos /= nStations;
  std::cout << "info: mean pos " << meanPos
            << " for " << nStations << " stations" << std::endl;
  for (auto& s : stationInfos) {
    if (s)
      s.fPosition -= meanPos;
  }

  return meanPos;
}


template<typename T>
double
GetDistanceToLine(const t2::Vector<T>& point,
                  double phi,
                  const t2::Vector<T>& startOfLine)
{
  const t2::Vector<T> directionOnGround(cos(phi), sin(phi), 0);
  const t2::Vector<T> deltaX0 = point - startOfLine;
  t2::Vector<T> crossOver = directionOnGround;
  crossOver *= (directionOnGround*deltaX0);
  crossOver.fX += startOfLine.fX;
  crossOver.fY += startOfLine.fY;

  crossOver.fX -= point.fX;
  crossOver.fY -= point.fY;

  return crossOver.XYMag();
}


//uses: a = (-750, 1299, 0); b = (1500, 0, 0) as grid vectors
template<typename T>
std::pair<int, int>
GetGridVectors(const t2::Vector<T>& delta)
{
  const int n = round(delta.fY/1299.);
  const int m = round((delta.fX + 750.*n)/1500.);
  //std::cout << "delta " << delta << " " << n << " " << m << std::endl;
  return std::make_pair(n, m);
}


//used in classifcation of patterns
bool
IsSmaller(const std::tuple<int, int, bool>& a,
          const std::tuple<int, int, bool>& b);


std::string
GenerateKey(const std::vector<std::tuple<int, int, bool>>& data);

#endif