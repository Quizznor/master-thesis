#ifndef _IntAnalyser_
#define _IntAnalyser_ 

#include <io/RootOutFile.h>
#include <T2Interferometry/TimeStamp.h>
#include <T2Interferometry/ProjectedPositions.h>
#include <T2Interferometry/InterferometricDataHandler.h>
#include <T2Interferometry/Utl.h>
#include <ROOT-utl/THealPix.h>

class InterferometricAnalyser
{
private:
  IDataHandler* fDataHandler;
  THealPixD fGalacticDirections;
  THealPixF fLocalSkyCoordinates;

  std::string foutBaseName;
  io::RootOutFile<TimeStamp> fOutPut;

  uint fThreshold = 3;
  double fMaxTimeSpreadPerT2 = 15;    //first guess to test things!
  uint fPrintSecond = 2000000001;
  void FindEvents(uint bin);
  void SaveEvent(TimeStamp&, const std::pair<double, double>& , const std::vector<double>&);

public:
  InterferometricAnalyser(std::string outBaseName);
  InterferometricAnalyser(std::string outBaseName, IDataHandler* h);
  ~InterferometricAnalyser();

  void SetDataHandler(IDataHandler* h) { fDataHandler = h; }
  void SetPrintSecond(uint s) { fPrintSecond = s; }
  void SetTimeSpreadThreshold(double t) { fMaxTimeSpreadPerT2 = t; }
  void Analyse(uint firstGPSs = 0, uint lastGPSs = 2000000000);
};

#endif