#ifndef XBT2_H
#define XBT2_H
#include "Xb.h"

#define NT2MAX 45000 /* >> 32000 */

/// a second level trigger
struct t2 {
  int time; /* is unsigned at the end, but I need it signed */
  unsigned short ID;
  unsigned short energy;
};

/// a list of t2
struct t2list {
  unsigned int nelem;
  const char* algo;
  struct t2 *point;
};

/// information for one station in a t3: id, offset, window
struct t3stat {
  short offset; /* is really short and not unsigned */
  unsigned short ID;
  unsigned short window;
};

/// a t3: algo, second, usecond and a vector of \<t3stat\>
struct t3 {
  std::vector<t3stat> point;
  unsigned int refSecond;
  int refuSecond; /* is unsigned at the end, but I need it signed */
  const char* algo;
};

#endif
