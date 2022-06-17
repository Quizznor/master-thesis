/*
	Designed to save data for Studies based on minutes/hour basis
*/
#ifndef _avgvalues_
#define _avgvalues_
#include <Rtypes.h>
#include <utl/Accumulator.h>
#include <sd/Constants.h>

typedef unsigned short ushort;
typedef unsigned int uint;


class AvgValues
{
public:
	float fRawMean[sd::kNStations][5];
	float fRawVar[sd::kNStations][5];

	float fMeanAoP[sd::kNStations];											//might be necessary for all data (unclear), now only scaler; gives a correction factor
	float fPMTBaseline[sd::kNStations][3];
	float fPeak[sd::kNStations][3];
	
	float fMeanPressure;
	float fMeanTemperature;

	bool fJumpFlag[sd::kNStations];
	bool fUnstable[sd::kNStations];
	bool fUnstableBaseline[sd::kNStations];
	bool fUnstableScaler[sd::kNStations];


	const uint fGPSsecondBegin;
	const uint fGPSsecondEnd;

	ushort fActiveSeconds[sd::kNStations][2];						//saves how many seconds were actually used to compute the averages (0: T2, 1: Scaler)

	AvgValues();
	AvgValues(const uint& GPSBegin, const uint& GPSEnd);

	float GetMean(const uint& channel, const uint& station) const;
	float GetVar(const uint& channel, const uint& station) const;

	uint GetBegin() const;
	uint GetEnd() const;

	//double GetCorrectedScalers(uint station) const;

	ClassDefNV(AvgValues, 1);
};

#endif