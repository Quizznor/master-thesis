#ifndef _AdstEvent_h_
#define _AdstEvent_h_

#include <iostream>


struct AdstEvent {
  unsigned long long fGPS = 0;
  long int fMicrosecond = 0;
  unsigned long long fEventId = 0;

  double fTheta = 0;
  double fPhi = 0;

  double fX = 0;
  double fY = 0;
  double fZ = 0;

  double fLgE = 0;

  bool
  operator<(const AdstEvent& e)
    const
  {
    if (fGPS < e.fGPS)
      return true;
    else if (fGPS == e.fGPS)
      return fMicrosecond < e.fMicrosecond;
    return false;
  }

  void
  FixTime()
  {
    while (fMicrosecond > 1e6) {
      fMicrosecond -= 1e6;
      ++fGPS;
    }

    while (fMicrosecond < 0) {
      fMicrosecond += 1e6;
      --fGPS;
    }
  }
};

inline
std::istream&
operator>>(std::istream& in, AdstEvent& e)
{
  return in >> e.fGPS >> std::ws >> e.fMicrosecond
            >> std::ws >> e.fEventId >> std::ws
            >> e.fTheta >> std::ws >> e.fPhi >> std::ws
            >> e.fX >> std::ws >> e.fY >> std::ws >> e.fZ >> std::ws
            >> e.fLgE;
}
/*
  output << sde.GetGPSSecond() << ' '
         << uint(sde.GetGPSNanoSecond() * 1e-3)
         << ' ' << sde.GetEventId() << ' '
         << theta << ' '
         << phi << ' '
         << core.x() << ' '
         << core.y() << ' '
         << core.z() << ' '
         << std::log10(sh.GetEnergy())
         << '\n';
*/
//800803465 176960 1376625 0.655278 1.75642 464431 6.091e+06 1411.33 17.9972
#endif