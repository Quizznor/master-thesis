#include <Rtypes.h>
#include <sd/Constants.h>

#ifndef _secondData_
#define _secondData_

typedef unsigned short ushort;
typedef unsigned int uint;
/*
	Stores and Handles the data from all 'triggers' of one GPS second.

*/
#ifdef __CINT__
#  define ARRAY2_INIT
#else
#  define ARRAY2_INIT = { { 0 } }
#endif

class SecondData
{
public:
	//channels: ToT, T2, calT (70Hz), T1, Scaler
	ushort fDataArrays[5][sd::kNStations] ARRAY2_INIT;	//T1 and calT should get a factor 10 to account for averaging over minutes
	bool fJumpFlag[sd::kNStations];

	ushort fNActive = 0;

	uint fGPSSecond = 0;

	int SetStationData(ushort* data, uint channel);
	void SetGPSsecond(uint);

	SecondData() {}
	SecondData(uint GPSs) : fGPSSecond(GPSs) {}
	~SecondData() {}

	int GetStationData(ushort* out, int id) const;
	int GetT2Data(ushort*) const;
	int GetToTData(ushort*) const;
	int GetT1Data(float*) const;				//returns the real value, as internally it's stored with a factor 10
	int GetCalTData(float*) const;			// -- '' '' --
	int GetScalerData(ushort*) const;

	int GetData(ushort* out, uint channel) const; //returns the internal values, e.g. averaged values are still multiplyied with a factor 10
	uint GetGPSsecond() const; 

	void SetNActive(uint nactive);
	uint GetNActive() const;

	explicit operator bool() const { return fNActive; }

	friend class SimDataCreator;
	friend class Converter;

	ClassDefNV(SecondData, 2);
	
};
#endif