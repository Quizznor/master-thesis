#ifndef _Units_h_
#define _Units_h_


namespace t2 {

  static bool useOldMetric = false;

  const double kMeter = 1;
  const double kMicroSecond = 300;
  const double kSecond = 1e6; //does this definition make sense?
  const double kOneSecond = kSecond;

  const double kPi = 3.14159265;
  const double kInvPi = 1/kPi;
  const double kOneUSec = kMicroSecond;    // alias
  const double kOneUSecSqr = utl::Sqr(kOneUSec);
  const double kTimeDistanceScale2 = 1./kOneUSecSqr; //us^s used to be 1./(1000.*1000.);

  const double kTimeVariance2 = kOneUSecSqr/12; //Var(uniform-distr.) = width/12.
                                                // note: is just overall scaling, with measured
                                                // sigma = 485 ns, change epsilon(db) to 0.53 and it's the same
  const int kJitter = 3;  //c.f. XbAlgo.h

  //for merging
  const double kToleranceDistance2 = 50*50./*m*/;
  const double kMaxTimeDifference = 50 * kMicroSecond/*us*/;  //also used by GraphSearch/GraphNode

  // is the default option, can be adjusted with
  //  GraphSearch::fMaxDeltaTLightning
  const double kMaxLightningSearchTimeDifference = 20 * kMicroSecond; //should be sufficient... factor 2 above c

}

#endif