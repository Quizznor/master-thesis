#ifndef _JumpOutput_
#define _JumpOutput_

#include <Rtypes.h>
#include <sd/Constants.h>
#include <vector>

typedef unsigned int uint;
typedef unsigned short ushort;

struct JumpOutput{
    std::vector<uint> fJumpTimes[sd::kNStations];
    std::vector<double> fJumpHeights[sd::kNStations];
    std::vector<double> fMerrit[sd::kNStations];   //as |mean1 - mean2|/sqrt(var_mean1 + var_mean2)
    ushort fModuloStation[sd::kNStations];         // for 61s based analysis: which GPSsecond % 61 == x is the right one.
    ClassDefNV(JumpOutput, 3);
};

#endif
