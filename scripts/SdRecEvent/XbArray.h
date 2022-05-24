#ifndef XBARRAY_H
#define XBARRAY_H

#define ROW (maxstat+1) // The maximum Id of the stations + 1
#define FAKE 0          // XXX Assuming no station has ID 0 XXX

struct Station {
  double Northing;
  double Easting;
  double Altitude;
  int Id;
};

extern char *neighbour;
extern char *distcrown;
extern char *exists;
extern struct Station* array;
extern int nstat;
extern int maxstat;

int crown(double, double, double, double);
void BuildNeighbour();
void XbArrayInit(int);

#define CARMEN 49
#define MIRANDA 64
#define DIA 139
#define NOCHE 186
#define MOULIN 140
#define ROUGE 185
#endif
