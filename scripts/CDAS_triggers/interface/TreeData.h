#ifndef _TreeData_
#define _TreeData_

#include <Rtypes.h>
#include <sd/LowEnergyData.h>

struct TreeData{
  uint fScaler = 0;
  uint fT2 = 0;
  uint fID = 0;
  uint fGPSSecond = 0;

  float fArea[3];
  float fPeak[3];
  float fPMV[3];
  float fDACPM[3];

  float f24V = 0;
  float f3V = 0;
  float f_3V = 0;
  float f12V = 0;
  float f5V = 0;
  float fSolarPanelV = 0;
  float fSolarPanelI = 0;
  float fBatteryV[2];

  float fPMTTemperature;
  //float fHighVoltage = 0;
  float fPMTCurrent = 0;
  
  float fADCBaseline = 0;
  //unsigned int fCal100 = 0;
  //unsigned int fCal40 = 0;
  float fVarianceDynode = 0;
  unsigned int fDeadTime = 0;
  float fCurrentLoad = 0;
  float fElectT = 0;

  float fHeight;

  float fTemperature = 0;       // deg C
  float fHumidity = 0;          // %
  float fAverageWindSpeed = 0;
  float fPressure = 0;          // in hPa
  float fPressure1450 = 0;    // hPa
  float fPressure2000 = 0;    // hPa
  float fTemperature1450 = 0; // C
  float fTemperature2000 = 0; // C

  /*void
  SetData(const sd::LowEnergyData& ev, uint id)
  {
    fScaler = ev.fStation[id].fScaler;
    fT2 = ev.fStation[id].fT2;
    fID = id;
    fGPSSecond = ev.fGPSSecond;

    for (uint i = 0; i < 3; ++i) {
      fArea[i] = ev.fStation[id].fArea[i];
      fPeak[i] = ev.fStation[id].fPeak[i];
      fPMV[i] = ev.fStation[id].fPMV[i];
      fDACPM[i] = ev.fStation[id].fDACPM[i];
      //fPMTTemperature[i] = ev.fStation[id].fPMTTemperature[i];
    }

    f24V = ev.fStation[id].f24V;
    f12V = ev.fStation[id].f12V;
    f5V = ev.fStation[id].f5V;
    f3V = ev.fStation[id].f3V;
    f_3V = ev.fStation[id].f_3V;

    fSolarPanelI = ev.fStation[id].fSolarPanelI;
    fSolarPanelV = ev.fStation[id].fSolarPanelV;
    fBatteryV[0] = ev.fStation[id].fBatteryV[0];
    fBatteryV[1] = ev.fStation[id].fBatteryV[1];

    //fArea = ev.fStation[id].fAreaOverPeak;
    //fHighVoltage = ev.fStation[id].fHighVoltage - HVInit;
    fElectT = ev.fStation[id].fElectT;
    fPMTCurrent = ev.fStation[id].fPMTCurrent;
    fPMTTemperature = ev.fStation[id].fPMTTemperature;
    fADCBaseline = ev.fStation[id].fADCBaseline;
    //fCal100 = ev.fStation[id].fCal100;
    //fCal40 = ev.fStation[id].fCal40;
    fVarianceDynode = ev.fStation[id].fVarianceDynode;
    fDeadTime = ev.fStation[id].fDeadTime;

    fCurrentLoad = ev.fStation[id].fCurrentLoad;

    fTemperature = ev.fAtmosphere.fTemperature;
    fHumidity = ev.fAtmosphere.fHumidity;
    fAverageWindSpeed = ev.fAtmosphere.fAverageWindSpeed;
    fPressure = ev.fAtmosphere.fPressure;
    fPressure1450 = ev.fAtmosphere.fGdas.fPressure1450;
    fPressure2000 = ev.fAtmosphere.fGdas.fPressure2000;
    fTemperature1450 = ev.fAtmosphere.fGdas.fTemperature1450;
    fTemperature2000 = ev.fAtmosphere.fGdas.fTemperature2000;
  }*/

  ClassDefNV(TreeData, 5);
};

#endif