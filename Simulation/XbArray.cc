/*! \file
    \brief Deals with crowns for the array of stations, building neighbour array
*/

#include "XbArray.h"
#include "Xb.h"
#include <cstdlib>
#include <cstdio>

char *neighbour=NULL;
char *distcrown=NULL;
char *exists=NULL;

int crown(double x1, double x2, double y1, double y2) {
  double d1=fabs(y1-y2)/2600+fabs(x1-x2)/1500;
  double d2=fabs(y1-y2)/1300;
  double max=(d1>d2 ? d1 : d2);
  int ret=(int)(floor(max));
  if (max-ret>0.2) ret++; // rounding at 0.2 crowns, ie 300 m
  return ret;
}

void BuildNeighbour() {
  if (neighbour) free(neighbour);
  if (exists) free(exists);
  if (distcrown) free(distcrown);
  neighbour=(char *)malloc(ROW*ROW*sizeof(char));
  distcrown=(char *)malloc(ROW*ROW*sizeof(char));
  exists=(char *)malloc(ROW*sizeof(char));
  // An unknown or FAKE station has no neighbour, so it will never trigger.
  for (int i=0;i<ROW;i++) {
    exists[i]=0;
    for (int j=0;j<ROW;j++) {
      neighbour[i*ROW+j]=0;
      distcrown[i*ROW+j]=99;
    }
  }
  // for the known ones
  for (int i=0;i<nstat;i++) {
    exists[array[i].Id]=1;
    for (int j=0;j<nstat;j++) {
      neighbour[array[i].Id*ROW+array[j].Id]=crown(array[i].Easting,array[j].Easting,array[i].Northing,array[j].Northing);
      distcrown[array[i].Id*ROW+array[j].Id]=crown(array[i].Easting,array[j].Easting,array[i].Northing,array[j].Northing);
      if(neighbour[array[i].Id*ROW+array[j].Id]>4) neighbour[array[i].Id*ROW+array[j].Id]=0;
    }
  }
  // DOUBLETS must not be neighbours because of high coincidence rate
  neighbour[CARMEN*ROW+MIRANDA]=neighbour[MIRANDA*ROW+CARMEN]=0;
  neighbour[DIA*ROW+NOCHE]=neighbour[NOCHE*ROW+DIA]=0;
  neighbour[MOULIN*ROW+ROUGE]=neighbour[ROUGE*ROW+MOULIN]=0;
  // CDAS IS GOING DOWN NOW!!!! XB - 08-07-08 --- Turned off 09-07-08
  // 710     Oye     Olentangy       1110
  // 1575    Vanesa  Olentangy       2020
  // 1618    Voldemort       Olentangy       2090
//#define OYE  710
//#define VANESA 1575
//#define VOLDEMORT 1618
  //neighbour[OYE*ROW+VANESA]=neighbour[VANESA*ROW+OYE]=1;
  //neighbour[VOLDEMORT*ROW+VANESA]=neighbour[VANESA*ROW+VOLDEMORT]=1;
  //neighbour[OYE*ROW+VOLDEMORT]=neighbour[VOLDEMORT*ROW+OYE]=1;
}
