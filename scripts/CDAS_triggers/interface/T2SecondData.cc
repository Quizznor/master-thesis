#include <interface/T2SecondData.h>

ClassImp(T2SecondData)

T2SecondData::T2SecondData(){
}

T2SecondData::~T2SecondData(){

}

std::vector<double>::iterator 
T2SecondData::Begin(uint station)
{
  if(station < 2000){
    return times[station].begin();
  } else {
    std::cerr << "wrong station requested" << std::endl;
    return times[0].begin();
  }
}

std::vector<double>::iterator
T2SecondData::End(uint station)
{
  if(station < 2000){
    return times[station].end();
  } else {
    std::cerr << "wrong station requested" << std::endl;
    return times[0].end();
  }
}

void 
T2SecondData::PushBack(double time, uint station)
{
  if(station < 2000){
    times[station].push_back(time);
  }
}

uint 
T2SecondData::GetNT2(uint station) const
{
  if(station < 2000)
    return times[station].size();
  else
    return 0;
}

void 
T2SecondData::Clear()
{
  for (int i = 0; i < 2000; ++i){
    times[i].clear();
  }
}
