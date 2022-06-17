#ifndef _T2seconddata_
#define _T2seconddata_

#include <vector>
#include <iostream>
#include <Rtypes.h>


class T2SecondData
{
public:
	std::vector<double> times[2000];

public:
	uint GPSsecond = 0;

	T2SecondData();
	~T2SecondData();

	void Clear();

	uint GetNT2(uint station) const;
	std::vector<double>::iterator Begin(uint station);
	std::vector<double>::iterator End(uint station);

	void PushBack(double time, uint station);

	ClassDefNV(T2SecondData, 1);
};
#endif