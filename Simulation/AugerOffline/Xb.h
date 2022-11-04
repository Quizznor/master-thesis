#ifndef XB_H
#define XB_H

#include <string>
#include <vector>
#include <iostream>
#include<sstream>
#include <math.h>

//#ifndef __DPA__
//#include "IkC.h"
//#else
#define InternalLog(A,B) \
std::cout << "#B : " << A << std::endl;
//#endif


extern int verbose,debug,trivial_algo,carmen_miranda,observer,newt2config;
extern unsigned int second_;

#endif
