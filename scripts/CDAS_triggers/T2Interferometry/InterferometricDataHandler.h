#ifndef _IDataHandler_
#define _IDataHandler_

#include <vector>
#include <string>
#include <utility>
#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <T2Interferometry/ProjectedPositions.h>
#include <interface/T2.h>

#define kNT2MAX 64000
/*
  Class handling T2 Data for interferometric analysis
  - projects T2s in NBis planes for each GPSsecond
  - sorts them w.r.t time (microseconds)
  - reads from binary files
*/

class IDataHandler
{
private:
  std::vector<std::vector<projectedT2> > fProjectedMicroSeconds; //stores the T2 data sorted w.r.t station ID
  ProjectedPositions fPositions;
  std::vector<std::string> fFilenames;                   //the actual data files opened for analysis (loop over them)
  unsigned fItFilename = 0;
  FILE* fFile;                           //Input-file for piping the binary file

  void SetInputFile(const char*);        //Changes Datafile used
  int ReadSecond();
  
  int ReadT2(T2 input[kNT2MAX]);
  void ResetVectors();
  

public:
  IDataHandler();
  IDataHandler(char*);
  IDataHandler(std::vector<std::string> filenames);
  ~IDataHandler();

  bool fEoF = false;
  unsigned int fGPSSecond = 0;               //current (first) GPS second stored in this DataHandler

  unsigned int fnT2 = 0;                     //total number of read in T2s
  unsigned int fnT2_reject = 0;              //total number of rejected T2s (corrupted or wrong Trigger ID (e.g. scalers))  

  void ReadNextSecond();

  uint GetNBins() { return fPositions.GetNBins(); }

  /*int 
  GetBin(double cosTheta, double phi) 
  { 
    return fPositions.GetBin(cosTheta, phi); 
  }*/

  std::pair<double, double> 
  GetBinCenter(unsigned binNumber)
  {
    return fPositions.GetBinCenter(binNumber);
  }

  void SetOrder(int order);

  /*void 
  SetNBins(unsigned nCosTheta, unsigned nPhi) 
  { 
    fPositions.SetNBins(nCosTheta, nPhi);
    fProjectedMicroSeconds.clear();
    std::vector<projectedT2> tmp;

    for (uint i = 0; i < nCosTheta*nPhi; ++i)
      fProjectedMicroSeconds.push_back(tmp);
  }*/

  std::vector<projectedT2>& operator[](unsigned i) 
  {
    return fProjectedMicroSeconds[i];
  }
  friend class InterferometricAnalyser;
};
#endif