#include <string>
#include <iostream>
#include <cstdlib>
#include <stdio.h>

#define NT2MAX 45000 // >> 32000


struct T2 {
  int fTime;                // time in microsecond of the t2
  unsigned short fId;       // station Id
  unsigned short fTriggers; // trigger flag, called "energy" in central trigger code for historical reasons, kept like that for compatibility
};
// Currently, trigger flag on most stations can be: 
//   1 (single threshold)
//   7 (scaler, meaning the "t2" is not a t2 but scaler data, should be ignored for trigger studies)
//   9 (time over threshold)
//
// Some stations have a modified trigger sending more flags:
//   8 (ToT)
//  10 (ToTD)
//  11 (MOPS)
/*  
	------------------------------------------------
	  Functions for reading the binary data files 
	------------------------------------------------	
*/
int
ReadSecond(FILE* f)
{
  int refSecond = 0;
  if (!feof(f) && fread((void*)&refSecond, sizeof(refSecond), 1, f))
    return refSecond;
  return 0;
}

int
ReadT2(FILE* f, T2 input[NT2MAX])
{
  int nT2 = 0;
  if (fread((void*)&nT2, sizeof(nT2), 1, f) &&
      fread((void*)input, sizeof(T2), nT2, f))
    return nT2;
  return 0;
}

//---------------------------------------------------------------------
//---------------------------------------------------------------------