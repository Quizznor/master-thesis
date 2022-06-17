#ifndef _T2_
#define _T2_

/*#include <T2Interferometry/ProjectedPositions.h>

//after applying projection to plane of some direction bin
struct projectedT2
{
  double fTime;             // time in microsecond of the t2
  unsigned short fId;       // station Id

  bool operator>(const projectedT2& b) { return fTime > b.fTime; }
  bool operator<(const projectedT2& b) { return fTime < b.fTime; }

  double
  GetDistance(const ProjectedPositions& pos, uint id2)
  {
    return pos.GetRealDistance(fId, id2);
  }
};*/

// Currently, trigger flag on most stations can be: 
  //   1 (single threshold)
  //   7 (scaler, meaning the "t2" is not a t2 but scaler data, should be ignored for trigger studies)
  //   9 (time over threshold)
  //
  // Some stations have a modified trigger sending more flags:
  //   8 (ToT)
  //  10 (ToTD)
  //  11 (MOPS)

struct T2 {
  int fTime = 0;                // time in microsecond of the t2
  unsigned short fId = 0;       // station Id
  unsigned short fTriggers = 0; // trigger flag, called "energy" in central trigger code for historical reasons, kept like that for compatibility

  bool operator>(const T2& b) const { return fTime > b.fTime; }
  bool operator<(const T2& b) const { return fTime < b.fTime; }
  
  bool IsToT() const { return fTriggers == 9 || fTriggers == 8; }

  void SubtractSecond() { fTime -= 1000000; }
  void AddSecond() { fTime += 1000000; }

  bool 
  operator==(const T2& b)
    const 
  {
    return fTime == b.fTime && fId == b.fId && fTriggers == b.fTriggers;
  }
  T2 () { }
  T2(const int time, const ushort id) : fTime(time), fId(id), fTriggers(9) { }
  T2(const int time, const ushort id, int type) : fTime(time), fId(id), fTriggers(type) { }

  template<class C>
  T2(const C& c) : fTime(c.fTime/300), fId(c.fId), fTriggers(c.fTriggers) { } 
  //~T2() { }
};

#endif
