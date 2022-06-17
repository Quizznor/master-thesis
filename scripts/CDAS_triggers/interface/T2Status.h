#ifndef _T2Status_
#define _T2Status_ 

#include <Rtypes.h>

typedef unsigned int uint;

struct T2Status
{
  const uint fGPSSecond = 0;
  uint fnT2 = 0;
  uint fnT2_rejected = 0;
  uint fnScaler = 0;

  uint fId[2000];
  uint fnWithT2 = 0;

  T2Status(uint gps) : fGPSSecond(gps) 
  { 
    for (int i = 0; i < 2000; ++i)
      fId[i] = 0;
  }
  T2Status()
  { 
    for (int i = 0; i < 2000; ++i)
      fId[i] = 0;
  }
  ~T2Status() {}

  void
  GetNWithData()
  {
    fnWithT2 = 0;
    for (int i = 0; i < 2000; ++i) 
      if (fId[i])
        ++fnWithT2;
  }

  ClassDefNV(T2Status, 1);
};


#endif  
