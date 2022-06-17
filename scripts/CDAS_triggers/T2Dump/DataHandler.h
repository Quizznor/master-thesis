#ifndef _DataHandler_
#define _DataHandler_

#include <vector>
#include <string>
#include <utility>
#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <io/RootOutFile.h>
#include <interface/T2Status.h>
#include <interface/T2.h>

#define kNT2MAX 64000
/*
  Class handling T2 Data for analysis
  - stores Data sorted w.r.t Station ID and times
  - saves the data for one GPS second at a time
  - able to store general information on the data sample, e.g. number of T2s or # of rejected T2s 
*/

class DataIterator;


class DataHandler {
private:
  const std::string fUnzip = "pbzip2 -dc";
  //const std::string fUnzip = "bzip2 -dc";
  std::vector<T2> fFirstSecond;          //stores the T2 data sorted w.r.t station ID
  //std::vector<T2> fSecondSecond;         //stores the T2 data sorted w.r.t station ID
  std::vector<T2> fData;                 
  unsigned fSecondBoundary = 0;
  io::RootOutFile<T2Status> fOutStatus;
  const static std::map<ushort, ushort> fGridType;

  bool fError = false;

  int fShiftMicroSecond[2000];           //for background tests, shift t2's differently for different stations in time

  std::vector<std::string> fFilenames;   //the actual data files opened for analysis (loop over them)
  unsigned fItFilename = 0;
  FILE* fFile = nullptr;                       //Input-file for piping the binary file

  void SetInputFile(const char* const);        //Changes Datafile used

  //std::pair<std::vector<T2>*, std::vector<T2>*> ReadNextSecond();  
                                         //Reads the data for the next GPS second, returns pointer to the arrays of T2"s
                                         // the amount of T2s -> -1 if eof

  int
  ReadSecond()
  {
    int refSecond = 0;
    if (!feof(fFile) && fread((void*)&refSecond, sizeof(refSecond), 1, fFile))
      return refSecond;
    return -1;
  }

  int
  ReadT2(T2 input[kNT2MAX])
  {
    uint nT2 = 0;
    if (fread((void*)&nT2, sizeof(nT2), 1, fFile)) {
      if (nT2 > kNT2MAX) {
        fError = true;
        std::cerr << nT2 << " ";
        throw std::out_of_range("Read too few bytes; file out of sync!");
      }
      if (fread((void*)input, sizeof(T2), nT2, fFile))
        return nT2;
    }      
    return 0;
  }

public:
  bool fEoF = false;
  DataHandler(const std::string& Outputfilename);
  DataHandler(const std::string& Outputfilename, const char* const filename);
  DataHandler(const std::string& Outputfilename, 
              const std::vector<std::string>& filenames, bool backgroundRandom);
  ~DataHandler();

  unsigned int fGPSSecond = 0;               //current (first) GPS second stored in this DataHandler

  bool IsSecond(uint index) const { return index >= fSecondBoundary; }

  int ReadNextSecond(int nPositionInVector);
  void ReadNextSecond();

  bool fUseRandomShifts = false;
  unsigned int fnT2 = 0;                     //total number of read in T2s
  unsigned int fnT2_reject = 0;              //total number of rejected T2s (corrupted or wrong Trigger ID (e.g. scalers))  

  T2&
  operator[](const unsigned i) 
  {
    return fData[i];
  }

  const T2&
  operator[](const unsigned i) const
  {
    return fData[i];
  }

  T2&
  at(const unsigned i) 
  {
    if (i < fData.size())
      return fData[i];
    else
      throw std::out_of_range("out of range");
  }

  const T2&
  at(const unsigned i) const
  {
     if (i < fData.size())
      return fData[i];
    else
      throw std::out_of_range("out of range");
  }

  //std::vector<T2>::iterator begin() { return fData.begin(); }
  std::vector<T2>::const_iterator begin() const { return fData.begin(); }
  
  std::vector<T2>::iterator end() { return fData.end();}
  std::vector<T2>::const_iterator end() const { return fData.end(); }

  unsigned size() const { return fData.size(); }

  //friend class DataIterator;  
};
#endif
