#ifndef _sdet_UUBDownsampleFilter_h_
#define _sdet_UUBDownsampleFilter_h_

#include <utl/Trace.h>
#include <utl/TimeDistribution.h>
#include <utl/AugerUnits.h>
#include <utl/Math.h>


namespace sdet {

  /**
    Functions to perform the downsample algorithm used for legacy triggers in the UUB firmware

    Besed on descriptions and reference code from Dave Nitz

    The filter will only slightly change the scale of the trace values by factor 2048/2059,
    since the normalization in the firmware uses a bit-shift instead of a division with the
    exact norm. Note that the filter does not produce an UB equivalent trace, this would
    require the division of the trace with an additional factor of 4 which is ommited to
    the preserve dynamic range.

    \author Darko Veberic
    \date 13 Nov 2020
  */

  namespace {

    // FIR coefficients from Dave Nitz's implementation in UUB firmware
    constexpr int kFirCoefficients[] = { 5, 0, 12, 22, 0, -61, -96, 0, 256, 551, 681, 551, 256, 0, -96, -61, 0, 22, 12, 0, 5 };
    // FIR normalization
    // in firmware an 11-bit right shift is used instead, ie 2048
    constexpr int kFirNormalizationBitShift = 11;
    //const int kFirNormalization = 2059;  // true norm of FIR
    //constexpr double kFirNormalization = (1 << kFirNormalizationBitShift);  // actually used
    constexpr double kUbSampling = 25*utl::nanosecond;
    constexpr int kADCSaturation = 4095;


    // enforce ADC saturation, both ways
    inline
    constexpr
    int
    Clip(const int i)
    {
      return std::max(0, std::min(i, kADCSaturation));
    }
 
  }

  // phase can be one only of { 0, 1, 2 }
  inline
  utl::TraceI
  UUBDownsampleFilter(const utl::TraceI& trace, const int phase = 1)
  {
    // input trace is assumed to have 8.333ns binning
    const int n = trace.GetSize();
    if (!n)
      return utl::TraceI(0, kUbSampling);
    const int m = utl::Length(kFirCoefficients);
    const int m2 = m / 2;
    std::vector<int> t;
    t.reserve(n + 2*m2);
    // pad front with the first trace values, but backwards
    for (int i = m2; i; --i)
      t.push_back(trace[i]);
    // copy the whole trace
    for (int i = 0; i < n; ++i)
      t.push_back(trace[i]);
    // pad back with the last trace values, but backwards
    for (int i = 1; i <= m2; ++i)
      t.push_back(trace[n-1-i]);
    const int n3 = n / 3;
    utl::TraceI res(n3, kUbSampling);
    for (int k = 0; k < n3; ++k) {
      auto& v = res[k];
      const int i = 3*k + phase;
      for (int j = 0; j < m; ++j)
        v += t[i + j] * kFirCoefficients[j];
      v >>= kFirNormalizationBitShift;
      v = Clip(v);
    }
    return res;
  }
}


#endif