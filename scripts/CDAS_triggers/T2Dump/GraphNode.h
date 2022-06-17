#ifndef _GraphNode_
#define _GraphNode_

#include <Rtypes.h>
#include <T2Dump/Utl.h>
#include <t2/T2Data.h>
#include <T2Dump/Units.h>

/*
  Basic class to implement a search for compact formation of triggers
  contains information about position and the T2 Message
  important method:
    IsConnected( ... )
     essentially defines what a 'compact' formation means, by defining,
     what distances in space-time are still 'compact' -> dubbed connected
     This method is used when building t2::Graph candidates from the t2s
     in the t2::GraphSearch class

*/
namespace t2 {

  struct GraphNode {
    int fX = 0;
    int fY = 0;

    //this time is the microsecond, relative to the GPS second of the graph,
    // containing this node. As this can cover the second boundary
    // it is possible that fTime < 0, whereas fTime > 1e6 should not happen.
    // A t2::MergedCandidate will convert an event time to a value > 0 & < 1e6
    // but will leave the gps second of the graph unchanged
    // -> the timing of this t2 is recoverable, but has to be taken care of
    // when comparing e.g. to (raw) events, in order to avoid saving the
    // gps second in this class as well
    int fTime = 0;  /*us (!)*/

    ushort fTriggers = 0;
    uint fId = 0;

    //avoid searching for this point in the
    // data vectors -> faster when adding noise, etc.
    int fIndexInT2Array = 0;    //!

    GraphNode() = default;
    GraphNode(const int id, const int x, const int y) :
      fX(x),
      fY(y),
      fTime(0),
      fTriggers(1),
      fId(id)
    { }

    GraphNode(const int time, const int id,
              const int triggers,
              const int x, const int y) :
      fX(x),
      fY(y),
      fTime(time),
      fTriggers(triggers),
      fId(id)
    { }

    GraphNode(const T2Data& t2) :
      fTime(t2.fTime/kMicroSecond),
      fTriggers(t2.fTriggers),
      fId(t2.fId)
    {
      // prevent unnoticed bugs ...
      if (t2.fTime / 300. != t2.fTime / 300)
        throw std::runtime_error("GraphNode(T2Data&): time not in meter?");
    }

    ~GraphNode() = default;

    GraphNode(const GraphNode&) = default;


    operator bool() const { return fTriggers; }

    int
    GetCrown(const GraphNode& b)
      const
    {
      return crown(fX, b.fX, fY, b.fY);
    }

    bool
    IsConnected(const GraphNode& b,
                const bool coherent = false,
                const int maxCrown = 4,
                const double maxDeltaTLightning = kMaxLightningSearchTimeDifference)
      const
    {
      const int deltaTime = abs(b.fTime - fTime)*kMicroSecond;
      if (deltaTime > kMaxTimeDifference)
        return false;

      const int CDAScrown = crown(fX, b.fX, fY, b.fY);

      if (CDAScrown > maxCrown)
        return false;

      // note: what about nearest neighbours for Sd-Rings?
      if (!CDAScrown)
        return false;

      if (deltaTime <= (5*CDAScrown + kJitter) * kMicroSecond)  //<= because > is used for rejection
        return true;

      if (coherent)
        return false;

      //10 us aims at sd rings, lightcone would be 5
      if (CDAScrown <= 2 && deltaTime <= maxDeltaTLightning)
        return true;

      return false;
    }

    bool
    IsPlaneFrontDoublet(const GraphNode& b)
      const
    {
      const int deltaTime = abs(b.fTime - fTime);
      if (deltaTime > 5)
        return false;

      return crown(fX, b.fX, fY, b.fY) == 1;
    }

    bool IsToT() const
    { return T2Data(0, 0, fTriggers).IsToT(); }

    bool IsWide() const
    { return T2Data(0, 0, fTriggers).IsWide(); }

    bool
    operator==(const GraphNode& b)
      const
    {
      return    fX == b.fX
             && fY == b.fY
             && fTime == b.fTime
             && fTriggers == b.fTriggers
             && fId == b.fId;
    }

    bool operator==(const T2Data& t2)
      const
    {
      return fId == t2.fId && fTime == t2.fTime/300. && fTriggers == t2.fTriggers;
    }


    bool operator>(const GraphNode& b) const { return fTime > b.fTime; }
    bool operator<(const GraphNode& b) const { return fTime < b.fTime; }

    ClassDefNV(GraphNode, 1);
  };

  inline
  std::ostream&
  operator<<(std::ostream& os, const GraphNode& n)
  {
    return os << "(" << n.fTime << ", " << n.fId << ", " << n.fTriggers << ")";
  }
};

#endif