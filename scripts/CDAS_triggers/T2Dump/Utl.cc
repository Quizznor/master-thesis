#include <T2Dump/Utl.h>
#include <utility>
#include <cmath>
#include <exception>
#include <boost/tuple/tuple.hpp>
#include <T2Interferometry/AugerUnits.h>
#include <ctime>
#include <iostream>
#include <TMath.h>
#include <algorithm>

double
ModifiedJulianDate(uint gpsSecond)
{
  uint unixSecond = 315964783 + gpsSecond;
  const int secPerDay = 24*60*60;
  return double(unixSecond) / secPerDay + 40587.0;
}


time_t
GetUnixSecond(int year, int month, int day, int hour, int minute, int second)
{
  tm breakdown = {
    second,    // tm_sec
    minute,    // tm_min
    hour,      // tm_hour
    day,       // tm_mday
    month-1,   // tm_mon
    year-1900, // tm_year
    0,         // tm_wday
    0,         // tm_yday
    0,         // tm_isdst
    0, 0
  };
  const time_t t1 = mktime(&breakdown);
  const time_t dgm = mktime(gmtime(&t1));
  return t1 - (dgm - t1);
}

uint
YMDtoGPSs(int year, int month, int day, int hour, int minute, int second)
{
  return GetUnixSecond(year, month, day, hour, minute, second) - 315964783;
}


double
GPSsecond2GMST(uint gpsSecond)
{
  /// a la CDAS
  //const UTCDateTime p = gpssecond;
  //const TimeStamp time0 = UTCDateTime(p.GetYear(), p.GetMonth(), p.GetDay()).GetTimeStamp();
  gpsSecond += 3600;    //adjust for timezone UTC + 1
  tm gpsAfterEpoch = {
    gpsSecond,    // tm_sec
    0,    // tm_min
    0,    // tm_hour
    6,    // tm_mday
    0,    // tm_mon
    80,   // tm_year
    0,    // tm_wday
    0,    // tm_yday
    0,    // tm_isdst
    0, 0
  };

  time_t unixSecond = mktime(&gpsAfterEpoch);

  const tm* gm = gmtime(&unixSecond);
  const uint gpsYMD = YMDtoGPSs(gm->tm_year + 1900, gm->tm_mon + 1, gm->tm_mday);

  gm = gmtime(&unixSecond);

  const double julianDay0 = ModifiedJulianDate(gpsYMD) + 2400000.5;
  const double tt = (julianDay0 - 2415020.0) / 36525.0;
  const double dgmst0 = (0.000001075*tt + 100.0021359)*tt + 0.276919398;
  const double gmst0 = (dgmst0 - int(dgmst0)) * 24.0;

  double gmstOut = gmst0 + (gm->tm_hour + gm->tm_min/60. + gm->tm_sec/3600.) * 366.25 / 365.25;

  return gmstOut;
}


double
CalculateNutationCorrection(const double julianDay)
{
  /// a la CDAS
  const  double tt = (julianDay - 2415020.0) / 36525.; // time (Julius century) from 1900

  const double llt = (0.000303*tt + 36000.7689)*tt + 279.6967;
  const double ldt = (-0.001133*tt + 481267.8831)*tt + 270.4342;
  const double mmt = (-0.000150*tt + 35999.0498)*tt + 358.4758;
  const double mdt = (0.009192*tt + 477198.8491)*tt + 296.1046;
  const double omegat = (0.002078*tt - 1934.1420)*tt + 259.1833;

  const double ll = fmod(llt, 360.);  // mean ecliptic longtitude of the sun (deg)
  const double ld = fmod(ldt, 360.);  // mean ecliptic longtitude of the moon(deg)
  const double mm = fmod(mmt, 360.);  // mean anomaly of the sun  (deg)
  const double md = fmod(mdt, 360.);  // mean anomaly of the moon (deg)

  double omega = fmod(omegat, 360.);  // ecliptic lg. of ascending node of moon
  if (omega < 0)
    omega += 360;

  const double depsit =
      (9.2100 + 0.00091*tt) * cos(omega*deg)
    + (0.5522 - 0.00029*tt) * cos(2*ll*deg)
    -  0.0904               * cos(2*omega*deg)
    +  0.0884               * cos(2*ld*deg)
    +  0.0216               * cos((2*ll + mm)*deg)
    +  0.0183               * cos((2*ld - omega)*deg)
    +  0.0113               * cos((2*ld + md)*deg)
    -  0.0093               * cos((2*ll - mm)*deg)
    -  0.0006               * cos((2*ll - omega)*deg);

  const double epsi =
    ((0.000000503*tt - 0.00000164)*tt - 0.0130125)*tt + 23.452294 +
    0.00256*cos(omega*deg);  //sol.epsi
  double nuthour = depsit / 3600 * cos(epsi*deg);
  return nuthour;
}


std::vector<double>
ConvertToRaDec(const std::vector<double>& p)
{
  const double kPi = 3.141592;

  if (p.size() != 3) {
    throw std::out_of_range("Incorrect number of input parameters");
  }

  const uint gpssec = p[0];

  const double julianDay = ModifiedJulianDate(gpssec);
  const double gmst      = GPSsecond2GMST(gpssec);
  double theta           = p[1];
  double phi             = p[2];

  const double latitude = -35.079251*deg;   //converted from Default reference point
  const double longitude = -69.501295*deg;  // in ProjectedPositions

  phi += 0.5*kPi;  //phi should be -phi
                   //(Auger pointing east, and going to north, west, etc)
  phi *= -1;       // change sign of phi

  //phi += kPi;

  theta = 0.5*kPi - theta;  //theta

  const double sinTheta = sin(theta);
  const double cosTheta = cos(theta);
  const double cosPhi = cos(phi);
  const double sinLatitude = sin(latitude);
  const double cosLatitude = cos(latitude);
  const double dec = asin(sinLatitude*sinTheta - cosLatitude*cosTheta*cosPhi);
  const double secDec = 1 / cos(dec);
  const double haSin = cosTheta * sin(phi) * secDec;
  const double haCos = (cosLatitude*sinTheta + sinLatitude*cosTheta*cosPhi) * secDec;

  const double ha = (haSin < 0) ? 2*kPi - acos(haCos) : acos(haCos);

  const double nutHour = CalculateNutationCorrection(julianDay);

  //local sideral time
  double lmst = fmod(gmst + nutHour, 24.)*15 + longitude/deg;

  //get the lmst between 0 and 360 degrees
  if (lmst > 0)
    lmst -= int(lmst/360)*360;
  else
    lmst += 360 - int(lmst/360)*360;

  double ra = lmst*deg - ha;

  if (ra < 0)
    ra += 2*kPi;
  if (ra > 2*kPi)
    ra -= 2*kPi;

  std::vector<double> result(2);
  if (-2*kPi <= ra && ra <= 2*kPi) {
    result[0] = ra;
    result[1] = dec;
  }

  return result;
}

std::vector<double>
GetGalacticCoordinates(const std::vector<double>& p)
{
  const double kPi = 3.141592;

  if (p.size() != 3) {
    throw std::out_of_range("incorrect number of arguments");
  }

  const std::vector<double> q = ConvertToRaDec(p);

  const double ra  = q[0];
  const double dec = q[1];

  const double raGalEq   = 282.86*deg;  //ra of the ascending node
  const double galLg2000 = 32.933*deg;  //
  const double eqGal     = 27.13*deg;   // pi/2-angle betweem eq and gal equator

  const double cosDec = cos(dec);
  const double sinDec = sin(dec);
  const double cosEqGal = cos(eqGal);
  const double galLatitude = asin(sinDec*sin(eqGal) - cosDec*sin(ra - raGalEq)*cosEqGal);
  const double secGalLatitude = 1 / cos(galLatitude);
  const double cosLon = cosDec*cos(ra - raGalEq) * secGalLatitude;
  const double sinLon = (cosDec*sin(ra - raGalEq)*sin(eqGal) + sinDec*cosEqGal) * secGalLatitude;

  double galLongitude = ((cosLon < 0 && sinLon < 0) || (cosLon > 0 && sinLon < 0)) ?
    2*kPi - acos(cosLon) + galLg2000 : acos(cosLon) + galLg2000;

  //  while(GLongitude>2*kPi) GLongitude -= 2.0*kPi;
  galLongitude = fmod(galLongitude, 2*kPi);
  if (galLongitude < 0)
    galLongitude += 2*kPi;

  std::vector<double> result(2);
  result[0] = galLongitude;
  result[1] = galLatitude;
  return result;
}


//split x2 into a part parallel to x1, and a part perpendicular in 2D (ignoring z)
std::pair<TVector3, TVector3>
Get2DBasis(const TVector3& x1, const TVector3& x2)
{
  const TVector3 par = (x1*x2)*x1;
  TVector3 perp;
  double xy = pow(x1.x(), 2) + pow(x1.y(), 2);
  perp.SetX(sqrt(pow(x1.y(), 2)/xy));
  perp.SetY(-sqrt(pow(x1.x(), 2)/xy));

  return std::make_pair(par, TVector3((perp*x2)*perp));
}

// DBScan algorithm for clustering. Pair is
//  theta, phi
double
DirectionalDistance(const std::pair<double, double>& p1,
                    const std::pair<double, double>& p2)
{
  return acos(sin(p1.first)*sin(p2.first)
         + cos(p1.first)*cos(p2.first)*cos(p1.second - p2.second));
}

//returns a vector of indices that are inside epsilon of p
std::vector<uint>
RangeQuery(const std::vector<std::pair<double, double> >& directions,
          double epsilon,
          const std::pair<double, double>& p,
          double (*distance)(const std::pair<double, double>&, const std::pair<double, double>&))
{
  std::vector<uint> outputIds;

  for (uint i = 0; i < directions.size(); ++i) {
    if (directions[i] == p)
      continue;
    if (distance(directions[i], p) < epsilon) {
      outputIds.push_back(i);
    }
  }

  return outputIds;
}

//the number of points per cluster (minPoints)
// is excluding the current one (so in the range [0, inf))
std::vector<int>
DBScan(const std::vector<std::pair<double, double> >& directions,
      double epsilon,
      uint minPoints,
      int& nCluster,
      int& nMaxPerCluster)
{
  std::vector<int> labels;
  //labels: 0 undefined, > 0: cluster number; -1: noise
  // labels the directions in the directions vector according to clusters
  labels.resize(directions.size(), 0);

  int labelCounter = 0;

  for (uint i = 0; i < directions.size(); ++i) {
    if (labels[i])
      continue;

    auto neighbours = RangeQuery(directions, epsilon, directions[i], DirectionalDistance);
    if (neighbours.size() < minPoints) {
      labels[i] = -1;
      continue;
    }
    labels[i] = ++labelCounter;

    //while loop, as the vector might be altered by adding additional points
    // to the cluster; while to make sure, that this is not a simple for loop
    // neighbours are the indeces of the triplets in directions that are
    // neighours of the triplet-direction directions[i]
    uint index = 0;
    while (index < neighbours.size()) {

      if (labels[neighbours[index]] < 0)
        labels[neighbours[index]] = labelCounter;
      else if (labels[neighbours[index]]) {
        ++index;
        continue;
      }

      const auto additionalNeighbours = RangeQuery(directions, epsilon, directions[neighbours[index]], DirectionalDistance);
      labels[neighbours[index]] = labelCounter;

      if (additionalNeighbours.size() < minPoints) {
        ++index;
        continue;
      }

      for (const auto& j : additionalNeighbours)
        neighbours.push_back(j);

      ++index;
    }
  }
  nCluster = labelCounter;
  nMaxPerCluster = 0;

  std::vector<int> nPerCluster;
  nPerCluster.resize(labelCounter, 0);

  for (const auto& x : labels) {
    if (x > 0)
      ++nPerCluster[x - 1];
  }

  if (labelCounter)
    nMaxPerCluster = *std::max_element(nPerCluster.begin(), nPerCluster.end());

  return labels;
}


//for euclidian distance in (u,v) space
double EuclidianDistance(const std::pair<double, double>& p1,
                         const std::pair<double, double>& p2)
{
  return sqrt(pow(p1.first - p2.first, 2) + pow(p1.second - p2.second, 2));
}


std::vector<int>
DBScanUV(const std::vector<std::pair<double, double> >& directions,
      double epsilon,
      uint minPoints,
      int& nCluster,
      int& nMaxPerCluster)
{
  std::vector<int> labels;
  //labels: 0 undefined, > 0: cluster number; -1: noise
  // labels the directions in the directions vector according to clusters
  labels.resize(directions.size(), 0);

  int labelCounter = 0;

  for (uint i = 0; i < directions.size(); ++i) {
    if (labels[i])
      continue;

    auto neighbours = RangeQuery(directions, epsilon, directions[i], EuclidianDistance);
    if (neighbours.size() < minPoints) {
      labels[i] = -1;
      continue;
    }
    labels[i] = ++labelCounter;

    //while loop, as the vector might be altered by adding additional points
    // to the cluster; while to make sure, that this is not a simple for loop
    // neighbours are the indeces of the triplets in directions that are
    // neighours of the triplet-direction directions[i]
    uint index = 0;
    while (index < neighbours.size()) {

      if (labels[neighbours[index]] < 0)
        labels[neighbours[index]] = labelCounter;
      else if (labels[neighbours[index]]) {
        ++index;
        continue;
      }

      const auto additionalNeighbours = RangeQuery(directions, epsilon, directions[neighbours[index]], EuclidianDistance);
      labels[neighbours[index]] = labelCounter;

      if (additionalNeighbours.size() < minPoints) {
        ++index;
        continue;
      }

      for (const auto& j : additionalNeighbours)
        neighbours.push_back(j);

      ++index;
    }
  }
  nCluster = labelCounter;
  nMaxPerCluster = 0;

  std::vector<int> nPerCluster;
  nPerCluster.resize(labelCounter, 0);

  for (const auto& x : labels) {
    if (x > 0)
      ++nPerCluster[x - 1];
  }

  if (labelCounter)
    nMaxPerCluster = *std::max_element(nPerCluster.begin(), nPerCluster.end());

  return labels;
}


//convert a pair of points (in the array) to r,theta of the corresponding
// straight line
std::pair<double, double>
GetRTheta(const TVector3& p1, const TVector3& p2, double x0, double y0)
{
  const TVector3 delta = p1 - p2;
  const double theta = TMath::Pi()/2. - atan(delta.y()/delta.x());
  const double yC = p2.y() - y0 - delta.y()/delta.x()*(p2.x() - x0);

  const double ctnMinusTanPhi = tan(theta) - tan(TMath::Pi()/2. - theta);
  const double r = fabs(yC)*sqrt(1./pow(ctnMinusTanPhi, 2)
                  + pow(tan(theta)/ctnMinusTanPhi, 2));

  return std::make_pair(r, theta);
}

//copying function from XbArray.cc to get number of crown...
int
crown(double x1, double x2, double y1, double y2)
{
  const double d1 = fabs(y1-y2)/2600 + fabs(x1-x2)/1500;
  const double d2 = fabs(y1-y2)/1300;
  const double max = (d1 > d2 ? d1 : d2);
  int ret = floor(max);

  if (max - ret > 0.2)
    ++ret; // rounding at 0.2 crowns, ie 300 m

  return ret;
}

std::string
GenerateKey(const std::vector<std::tuple<int, int, bool>>& data)
{
  std::string output = std::to_string(data.size());
  for (const auto& t : data) {
    output += " (" + std::to_string(std::get<0>(t)) + " "
              + std::to_string(std::get<1>(t)) + " "
              + std::to_string(std::get<2>(t)) + ")";
  }
  return output;
}

bool
IsSmaller(const std::tuple<int, int, bool>& a,
          const std::tuple<int, int, bool>& b)
{
  if (a == b)
    return false;
  const int sumA = abs(std::get<0>(a)) + abs(std::get<1>(a));
  const int sumB = abs(std::get<0>(b)) + abs(std::get<1>(b));
  if (sumA < sumB)
    return true;
  if (sumA == sumB)
    return abs(std::get<1>(a)) < abs(std::get<1>(b));
  return false;
}
