#include <T2Interferometry/InterferometricDataHandler.h>
#include <algorithm>
#include <iostream>

IDataHandler::IDataHandler()
{
  for (uint i = 0; i < fPositions.GetNBins(); ++i) {
    std::vector<projectedT2> tmp;
    fProjectedMicroSeconds.push_back(tmp);
  }  
}

IDataHandler::~IDataHandler()
{
  pclose(fFile);
}

IDataHandler::IDataHandler(char* file)
{
  for (uint i = 0; i < fPositions.GetNBins(); ++i) {
    std::vector<projectedT2> tmp;
    fProjectedMicroSeconds.push_back(tmp);
  }

  char cmd[1000] = { '\0' };
  snprintf(cmd, 1000, "pbzip2 -dc %s", file);
  fFile = popen(cmd, "r");

  ReadNextSecond();
}

IDataHandler::IDataHandler(std::vector<std::string> filenames)
{
  for (uint i = 0; i < fPositions.GetNBins(); ++i) {
    std::vector<projectedT2> tmp;
    fProjectedMicroSeconds.push_back(tmp);
  }

  fFilenames = filenames;

  char cmd[1000] = { '\0' };
  snprintf(cmd, 1000, "pbzip2 -dc %s", filenames.front().c_str());
  fFile = popen(cmd, "r");

  fItFilename = 0;

  ReadNextSecond();
}


void 
IDataHandler::SetInputFile(const char* file) 
{
  pclose(fFile);

  char cmd[1000] = { '\0' };
  snprintf(cmd, 1000, "pbzip2 -dc %s", file);
  fFile = popen(cmd, "r");
}


int
IDataHandler::ReadSecond()
{
  int refSecond = 0;
  if (!feof(fFile) && fread((void*)&refSecond, sizeof(refSecond), 1, fFile))
    return refSecond;
  return 0;
}

int
IDataHandler::ReadT2(T2 input[kNT2MAX])
{
  int nT2 = 0;
  if (fread((void*)&nT2, sizeof(nT2), 1, fFile) &&
      fread((void*)input, sizeof(T2), nT2, fFile))
    if (nT2 > kNT2MAX)
      throw std::out_of_range("Read too few bytes for T2 data; file out of sync");
    return nT2;
  return 0;
}

void
IDataHandler::ResetVectors()
{
  for (uint i = 0; i < fProjectedMicroSeconds.size(); ++i) {
    fProjectedMicroSeconds[i].clear();
  }
}

void
IDataHandler::ReadNextSecond()
{
  ResetVectors();

  if (!feof(fFile)) {
    fEoF = false;
    T2 input[kNT2MAX];

    const int second = ReadSecond();
    if (!second) {
      if (fItFilename < fFilenames.size() - 1) {
        ++fItFilename;
        SetInputFile(fFilenames[fItFilename].c_str());

        return ReadNextSecond();
      } else {
        fEoF = true;
        return;  
      }
    }
    fGPSSecond = second;

    const int nT2 = ReadT2(input);
    // Data for one GPS second in second, and input (needs nT2 to avoid access violation)
    for (int i = 0; i < nT2; ++i) {
      const T2& t2 = input[i];

      if (t2.fTriggers == 7) {    //reject scaler data, not a real T2
        ++fnT2_reject;
        continue;
      }
      if (!fPositions.IdExists(t2.fId)) {
        std::cout << "warning, position to id " << t2.fId << " unknown" << std::endl;
        continue;
      }       
      projectedT2 pt2;
      pt2.fId = t2.fId;

      //checks for Data-quality
      if (t2.fTime >= 0 
          && t2.fTime < 1000000 
          && t2.fId <= 1999 
          && t2.fId > 0 
          && t2.fTriggers > 0 
          && t2.fTriggers < 15) { 
        
        for (uint i = 0; i < fPositions.GetNBins(); ++i) {
          pt2.fTime = t2.fTime + fPositions[t2.fId][i];

          fProjectedMicroSeconds[i].push_back(pt2);
        }
        ++fnT2;
      } else {
        ++fnT2_reject;
      }
    }//for loop over T2s in this second
    for (uint i = 0; i < fPositions.GetNBins(); ++i)
      std::sort(fProjectedMicroSeconds[i].begin(), fProjectedMicroSeconds[i].end());
  }
}

void
IDataHandler::SetOrder(int order)
{
  fPositions.SetOrder(order);
  fProjectedMicroSeconds.clear();

  std::vector<projectedT2> tmp;

  for (uint i = 0; i < fPositions.GetNBins(); ++i)
    fProjectedMicroSeconds.push_back(tmp);
}