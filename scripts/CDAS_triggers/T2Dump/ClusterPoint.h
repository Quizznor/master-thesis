#ifndef _ClusterPoint_h_
#define _ClusterPoint_h_

#include <T2Dump/Triplet.h>
#include <T2Dump/Units.h>


namespace t2 {
  
  struct ClusterPoint{
    float fu = 0;
    float fv = 0;
    double ft0 = 0;  //us!

    int fIds[3] = { 0, 0, 0 };

    ClusterPoint() { }
    ClusterPoint(const ClusterPoint& c) :
      fu(c.fu), fv(c.fv), ft0(c.ft0), fIds{c.fIds[0], c.fIds[1], c.fIds[2]} { }
    ClusterPoint(const rTriplet& t) : 
      fu(t.fu), fv(t.fv), ft0(t.ft0),
      fIds{t.fTrigger[0].fId, t.fTrigger[1].fId, t.fTrigger[2].fId} 
    { }

    double Distance2(const ClusterPoint& b) const
    {
      return utl::Sqr(fu - b.fu) + utl::Sqr(fv - b.fv) 
             + kTimeDistanceScale2*utl::Sqr(ft0 - b.ft0);
    }

    bool operator==(const rTriplet& r) const
    {
      return r.fu == fu && r.fv == fv && r.ft0 == ft0;
    }

    bool operator<(const ClusterPoint& b) const
    { return ft0 < b.ft0; }

    bool operator>(const ClusterPoint& b) const
    { return ft0 > b.ft0; }

    ClassDefNV(ClusterPoint, 2);
  };
};
#endif