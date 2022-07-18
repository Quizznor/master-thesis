#ifndef XBALGO_H
#define XBALGO_H
#include "XbT2.h"

/* error allowed in microseconds on the difference of time of two station */
/* 2 is enough. To be safe let's put 3 */
const int dtime=3;

// Binning is now dynamically adapted to number of T2s.
// If more than MAXSTATINONEBIN station fall in one bin, a special event is built
#define MAXBINNUMBER 110000
#define MAXSTATINONEBIN 30
/* maximum of T2 overlapping (in the 50us window) */
#define MAXOVERLAP 1024
// maximum of T2 from Carmen and Miranda
#define MAXCM 1024
// maximum of TOT: 10Hz*1600 and TOT flag in T2s (9)
#define MAXTOT 16000
// TOT is now >=8, scaler is 7, threshold is <=6
//
// values used 28/10/2016:
// 1: single threshold
// 2: wide threshold
// 7: scaler
// 8: large saturation
// 9: ToT
// 10: wide ToT
// 11: ToTd
// 12: wide ToTd
// 13: MOPS
// 14: wide MOPS (and wide ToTd + MOPS)
// 15: ToTd+MOPS
#define TOTVALUE 8

std::vector<t2list> XbAlgoT2Process(struct t2 *input,int nt2);
void XbAlgoT2Init();
void XbAlgoFree(struct t2list in);
t3 XbAlgoT3Build(struct t2list in);

#endif
