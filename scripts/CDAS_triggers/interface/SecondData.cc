#include <interface/SecondData.h>
#include <stdexcept>
#include <iostream>

ClassImp(SecondData)

int 
SecondData::SetStationData(ushort* Data, uint channel)
{
  if(channel >= 5)
    throw std::out_of_range("No such channel");

  for (int i = 0; i < sd::kNStations; ++i){
    fDataArrays[channel][i] = Data[i];
  }

  return 0;
}

void 
SecondData::SetGPSsecond(uint t)
{
  fGPSSecond = t;
}

int 
SecondData::GetStationData(ushort* out, int id) const
{
  if(id > sd::kNStations)
    throw std::out_of_range("No such station");

  try {
    for (int i = 0; i < 5; ++i){
      out[i] = fDataArrays[i][id];
    }
  } catch (std::exception& e) {
    std::cerr << "Exception while writing data: " << e.what() << std::endl;    
  }
  return 0;
  
}

int 
SecondData::GetT2Data(ushort* out) const
{
  try{
    for (int i = 0; i < sd::kNStations; ++i)
      out[i] = fDataArrays[1][i];
  } catch (std::exception& e) {
    std::cerr << "Exception writing Data: " << e.what() << std::endl;
  }
  return 0;
}

int 
SecondData::GetToTData(ushort* out) const
{
  try{
    for (int i = 0; i < sd::kNStations; ++i)
      out[i] = fDataArrays[0][i];
  } catch (std::exception& e) {
    std::cerr << "Exception writing Data: " << e.what() << std::endl;
  }
  return 0;
}

int 
SecondData::GetT1Data(float* out) const
{
  try{
    for (int i = 0; i < sd::kNStations; ++i)
      out[i] = fDataArrays[3][i];
  } catch (std::exception& e) {
    std::cerr << "Exception writing Data: " << e.what() << std::endl;
  }

  return 0;
}

int 
SecondData::GetCalTData(float* out) const
{
  try{
    for (int i = 0; i < sd::kNStations; ++i)
      out[i] = fDataArrays[2][i];
  } catch (std::exception& e) {
    std::cerr << "Exception writing Data: " << e.what() << std::endl;
  }

  return 0;
}

int 
SecondData::GetScalerData(ushort* out) const
{
  try{
    for (int i = 0; i < sd::kNStations; ++i)
      out[i] = fDataArrays[4][i];
  } catch (std::exception& e) {
    std::cerr << "Exception writing Data: " << e.what() << std::endl;
  }
  
  return 0;
}

int 
SecondData::GetData(ushort* out, uint channel) const
{
  if(channel > 4)
    throw std::out_of_range("No such channel!");
  
  for (int i = 0; i < sd::kNStations; ++i)
    out[i] = fDataArrays[channel][i];

  return 0;
}

uint 
SecondData::GetGPSsecond() const
{
  return fGPSSecond;
}

void
SecondData::SetNActive(uint nactive)
{
  fNActive = nactive;
}

uint
SecondData::GetNActive() const
{
  return fNActive;
}