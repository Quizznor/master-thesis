/*! \file
    \brief Main trigger algorithms
*/
#include "XbAlgo.h"
#include "XbArray.h"
#include <cstdlib>
#include <cstring>

using std::vector;
using std::stringstream;

#define FREE(_x_) { free(_x_); _x_ = nullptr; }

/// t2 at the end of the second that will be used in both seconds
static struct t2 extrat2[MAXOVERLAP], extrat2old[MAXOVERLAP];
/// t2 from one of the doublets to be used in the doublet special trigger
static struct t2 carmenmiranda[MAXCM];
/// the fake trigger, not in cluster with any trigger, to be used when one removes a trigger from a cluster
static struct t2 fake;
static int nextra = 0, delay = 0, delaycm = 0, delaymr = 0, delaydn = 0;
/// number of t2 per time bin
static unsigned short ncompteurt2[MAXBINNUMBER],ncompteurtot[MAXBINNUMBER];
/// histogram of these t2 per time bin, poiting to the precise t2
static struct t2 *compteurt2[MAXBINNUMBER][MAXSTATINONEBIN],*compteurtot[MAXBINNUMBER][MAXSTATINONEBIN];
static int binnumber = MAXBINNUMBER, oldbinnumber = MAXBINNUMBER, binsize;
static int cour[5];
/// number of t2 cluster
static int nbt2cluster;
/// t2 clusters
static struct t2list t2cluster[1000];
/// output of the algorithm: a list of clusters
vector < t2list > out;

int observer = 0;               // To request a specific station

/// initialisation of XbAlgoT2 algorithm, building fake station and resetting histograms
void XbAlgoT2Init() {
  fake.ID = FAKE;
  fake.time = 1100000;
  for (int j = 0; j < MAXBINNUMBER; j++)
    ncompteurt2[j] = ncompteurtot[j] = 0;
}

/// Full Array current trigger: 2C1 & 3C2 & 4C4
int XbAlgoT3FullArray() {
  return ((cour[1] > 0) && (cour[1] + cour[2] > 1)) && (cour[1] + cour[2] + cour[3] + cour[4] > 2);
}

/// ToT current trigger: 2C1 & 3C2
int XbAlgoT3ToT() {
  return ((cour[1] > 0) && (cour[1] + cour[2] > 1));
}

/// Function called when more than MAXSTATINONEBIN are in 5 consecutive bins of the time histogram
void HugeShower(struct t2 *input, int nt2, int msec) {
  int nhugestat=0;
  // keeping all stations within 100 us to be safe
  for (int j = 0; j < nt2; j++)
    if ((input[j].time > msec - 100) && (input[j].time < msec + 100))
      nhugestat++;
  if (! nhugestat)
    return;
  // building the event
  t2cluster[nbt2cluster].nelem = nhugestat;
  t2cluster[nbt2cluster].algo = "BIG";
  t2cluster[nbt2cluster].point = (struct t2 *) malloc(nhugestat * sizeof(struct t2));
  int m=0;
  for (int j = 0; j < nt2; j++)
    if ((input[j].time > msec - 100) && (input[j].time < msec + 100)) {
      *(t2cluster[nbt2cluster].point + m) = input[j];
      m++;
    }
  if (nbt2cluster<999) nbt2cluster++;
}

/// Function called to find clusters
void XbAlgoFindCluster(int minstat,struct t2 *input, int nt2, unsigned short *nbt2inbin, struct t2 *listt2inbin[MAXBINNUMBER][MAXSTATINONEBIN],int (*T3func)(),const char *algo) {
  int i,j,k,l,m;
  int index,nstat;
  int total = nbt2inbin[0] + nbt2inbin[1] + nbt2inbin[2] + nbt2inbin[3] + nbt2inbin[4];
  static struct t2 *liste[100];

  // j: center of the sliding bin-2 -> bin+2 window
  for (j = 2; j < binnumber - 2; j++) {
    // if we get enough stations in the sliding window, and there are stations in the central bin
    if (total >= minstat && nbt2inbin[j] > 0) {
      // if there is more than MAXSTATINONEBIN in the time bins
      // (is here to protect from memory violation if it happened, for big showers)
      if (total >= MAXSTATINONEBIN) {
        if (input && j>5) HugeShower(input, nt2, (j-5) * binsize);
      } else {
        /* for all the stations in the middle of the window */
        for (k = 0; k < nbt2inbin[j]; k++) {
          cour[0] = cour[1] = cour[2] = cour[3] = cour[4] = 0;
          /* counting the neighbour in each crown for all the stations  in the window */
          for (l = -2; l < 3; l++)
            for (m = 0; m < nbt2inbin[j + l]; m++)
              cour[(int) neighbour[listt2inbin[j][k]->ID * ROW + (int) listt2inbin[j + l][m]->ID]]++;
          if (T3func()) { // if the cluster is Ok
            /* we have to remove stations too far away in time */
            /* we do it now to optimise */
            /* now we will use lists of pointers */
            index = 0;
            for (l = -2; l < 3; l++)
              for (m = 0; m < nbt2inbin[j + l]; m++)
                liste[index++] = listt2inbin[j + l][m];
            /* 5us per crown + dtime jitter */
            for (l = 0; l < total; l++) {
              if (abs(listt2inbin[j][k]->time - liste[l]->time)
                  > 5 * neighbour[listt2inbin[j][k]->ID * ROW + liste[l]->ID] + dtime) {
                // station not time compatible => 1 less neighbour, replaced by fake station
                cour[(int) neighbour[listt2inbin[j][k]->ID * ROW + (int) liste[l]->ID]]--;
                liste[l] = &fake;
              }
            }
            nstat = cour[1] + cour[2] + cour[3] + cour[4];
            /* we have to mark stations with 2 t2 in the same cluster:
             * they only count as one, but both are kept                */
            for (i = 0; i < total; i++) {
              for (l = i + 1; l < total; l++) {
                if (liste[i]->ID == liste[l]->ID) {
                  cour[(int) neighbour[listt2inbin[j][k]->ID * ROW + (int) liste[l]->ID]]--;
                }
              }
            }
            if (T3func()) { // check again the cluster once bad timing stations have been removed
              // found a new cluster: building it
              nstat++;          /* + the central one */
              t2cluster[nbt2cluster].nelem = nstat;
              t2cluster[nbt2cluster].algo = algo;
              t2cluster[nbt2cluster].point = (struct t2 *) malloc(nstat * sizeof(struct t2));
              m = 0;
              for (l = 0; l < total; l++) {
                if (neighbour[listt2inbin[j][k]->ID * ROW + liste[l]->ID] > 0) {
                  *(t2cluster[nbt2cluster].point + m) = *liste[l];
                  m++;
                }
              }
              *(t2cluster[nbt2cluster].point + m) = *listt2inbin[j][k];
              if (nbt2cluster<999) nbt2cluster++;
            }
          }
        }
      }
    }
    // sliding window: add new bin, remove old one.
    total += nbt2inbin[j + 3] - nbt2inbin[j - 2];
    nbt2inbin[j - 2] = 0;
  }
}

/// Main function, processing T2 and returning clusters of T2
vector < t2list > XbAlgoT2Process(struct t2 *input, int nt2) {
  static struct t2 min;
  int nt2cm = 0, ntot = 0;
  int i, j, l;
  unsigned int k, m;
  int mink, supp, tcm, tdn, tmr, nbstat;
  static struct t2list t2mergedclusters[1000];

  out.clear();
  nbt2cluster=0;
  if (nt2 <= 0)
    return out;
  // upgrading configuration if needed
  if (newt2config) {
    delay = trivial_algo / 10;
    delaymr = delaydn = delaycm = carmen_miranda / 10;
    newt2config = 0;
  }
  // fill the histogram with >13 us bins
  // WARNING, there is a 50 usec covering with previous second (actually, 5 bins=5*binsize us)
  // bin size of the histogram is computed dynamically to be fast also when there are few t2s
  // binsize = 500000 / nt2;
  // if (binsize > 1000) binsize = 1000;
  // else if (binsize < 12) binsize = 12; // To get 2 (bins) * 12 (binsize) = 24 us > 20 (4 crowns) + 3 (jitter)
  binsize = 12; // fixed size now
  oldbinnumber = binnumber;
  binnumber = 1110000 / binsize;
  // filling the histogram with t2 from the previous second
  if (nextra > MAXOVERLAP) {
    stringstream s;
    s << "Too many extra triggers: " << nextra;
    InternalLog(s.str().c_str(),IKWARNING);
    nextra = MAXOVERLAP;
  }
  for (j = 0; j < nextra; j++) {
    extrat2old[j].time = extrat2[j].time - 1000000;     /* -1000000 as it is from previous second */
    extrat2old[j].ID = extrat2[j].ID;
    extrat2old[j].energy = extrat2[j].energy;
    k = 5 + extrat2old[j].time / binsize;       /* +5 for the 50us copy */
    if (5 + extrat2old[j].time / binsize < 0) {
      stringstream s;
      s << "Negative new time: " << 5 + extrat2old[j].time / binsize;
      InternalLog(s.str().c_str(),IKWARNING);
      k=0;                               /* could happen if bin size change dramatically from one second to the other one */
    }
    if (debug) {
      stringstream s;
      s << "extra t2:" << extrat2old[j].time << " -> " << k;
      InternalLog(s.str().c_str(),IKDEBUG);
    }
    if ((l = ncompteurt2[k]++) < MAXSTATINONEBIN)
      compteurt2[k][l] = &extrat2old[j];
    if ((extrat2old[j].ID == CARMEN) || (extrat2old[j].ID == MIRANDA)
        || (extrat2old[j].ID == MOULIN) || (extrat2old[j].ID == ROUGE)
        || (extrat2old[j].ID == DIA) || (extrat2old[j].ID == NOCHE))
      carmenmiranda[nt2cm++] = extrat2old[j];
    if (extrat2old[j].energy >= TOTVALUE)
      if ((l = ncompteurtot[k]++) < MAXSTATINONEBIN)
        compteurtot[k][l] = &extrat2old[j];
  }
  // filling histogram with all new t2
  nextra = 0;
  for (j = 0; j < nt2; j++) {
    if (input[j].time > 1000000) {
      stringstream s;
      s << "Time too big: " << input[j].time << " from LS " << input[j].ID;
      InternalLog(s.str().c_str(),IKERROR);
    }
    if (input[j].ID<100) continue; // WARNING!!!! Removing infill and doublets here. was is XbPm before!!!!!
    if (input[j].time > 1000000) continue;    /* should never happen */
    k = 5 + input[j].time / binsize;    /* +5 for the 50us copy (it's 5 bins) */
    // keeping t2 from the end of the second for next second
    if (input[j].time > 1000000 - 5*binsize) {
      extrat2[nextra++] = input[j];
      if (nextra >= MAXOVERLAP) {
        stringstream s;
        s << "Too many new extra triggers: " << nextra << ", removing 100";
        InternalLog(s.str().c_str(),IKERROR);
        nextra = MAXOVERLAP-100;
      }
    }
    if ((l = ncompteurt2[k]++) < MAXSTATINONEBIN)
      compteurt2[k][l] = &input[j];
    // filling special doublet trigger list
    if ((input[j].ID == CARMEN) || (input[j].ID == MIRANDA) ||
        (input[j].ID == MOULIN) || (input[j].ID == ROUGE) || (input[j].ID == DIA) || (input[j].ID == NOCHE))
      carmenmiranda[nt2cm++] = input[j];
    // filling tot t2 histogram
    if (input[j].energy >= TOTVALUE) {
      if ((l = ncompteurtot[k]++) < MAXSTATINONEBIN)
        compteurtot[k][l] = &input[j];
      ntot++;
    }
  }
  if (verbose) {
    stringstream s;
    s << "TOT triggers: " << ntot;
    InternalLog(s.str().c_str(),IKDEBUG);
  }
  /* Find clusters with Full Array algorithm */
  XbAlgoFindCluster(4,input,nt2,ncompteurt2,compteurt2,&XbAlgoT3FullArray,"2C1&3C2&4C4");
  /* Find clusters with TOT algorithm */
  XbAlgoFindCluster(3,NULL,0,ncompteurtot,compteurtot,&XbAlgoT3ToT,"TOT2C1&3C2");
  // cleaning
  for (j = binnumber - 5; j < oldbinnumber; j++)
    ncompteurt2[j] = ncompteurtot[j] = 0;
  // we have proto-clusters. Now, let's look at them and see if they have triggers in common for merging
  for (j = 0; j < nbt2cluster; j++) {
    // first, a t2mergedcluster is a t2cluster ordered in time
    l = 0;
    t2mergedclusters[j].nelem = t2cluster[j].nelem;
    t2mergedclusters[j].algo = t2cluster[j].algo;
    t2mergedclusters[j].point = (struct t2 *) malloc(t2mergedclusters[j].nelem * sizeof(struct t2));
    for (k = 0; k < t2cluster[j].nelem; k++) {
      min = fake;
      mink = 0;
      for (m = 0; m < t2cluster[j].nelem; m++) {
        if (((*(t2cluster[j].point + m)).time) < min.time) {
          min = *(t2cluster[j].point + m);
          mink = m;
        }
      }
      *(t2mergedclusters[j].point + k) = min;
      (*(t2cluster[j].point + mink)).time = 1010000;
    }
    // t2cluster is no more needed
    FREE(t2cluster[j].point);
  }
  /* Now, let's check for mixed clusters
   * There must be less than 100 us between any 2 stations */
  for (j = 0; j < nbt2cluster; j++) {
    for (i = j + 1; i < nbt2cluster; i++) {
      //if (abs((*(t2mergedclusters[j].point)).time - (*(t2mergedclusters[i].point)).time) < 100) {
      if ((t2mergedclusters[i].nelem>0 && t2mergedclusters[j].nelem>0) && (abs((*(t2mergedclusters[j].point)).time - (*(t2mergedclusters[i].point)).time) < 100)) {
        supp = 0;
        for (k = 0; k < t2mergedclusters[j].nelem; k++) {
          for (m = 0; m < t2mergedclusters[i].nelem; m++) {
            if (((*(t2mergedclusters[j].point + k)).ID == (*(t2mergedclusters[i].point + m)).ID)
                && ((*(t2mergedclusters[j].point + k)).time == (*(t2mergedclusters[i].point + m)).time)) {
              supp = 1;
            }
          }
        }
        if (supp == 1) {
          if (debug) {
        stringstream s;
            s << "supperposition of " << i << " and " << j;
        InternalLog(s.str().c_str(),IKDEBUG);
          }
          /* we build one out of two */
          /* we'll keep 'j' and remove 'i'. Change 'j' algo in case to keep "TOT" */
          if (strstr(t2mergedclusters[i].algo,"TOT")) t2mergedclusters[j].algo=t2mergedclusters[i].algo;
          for (m = 0; m < t2mergedclusters[i].nelem; m++) {
            supp = 0;
            for (k = 0; k < t2mergedclusters[j].nelem; k++) {
              if (((*(t2mergedclusters[j].point + k)).ID == (*(t2mergedclusters[i].point + m)).ID)
                  && ((*(t2mergedclusters[j].point + k)).time == (*(t2mergedclusters[i].point + m)).time)) {
                supp = 1;
              }
            }
            if (supp == 0) {
              /* t2mergedclusters[i].point+m is not in list[j] */
              t2mergedclusters[j].nelem++;
              t2mergedclusters[j].point = (struct t2 *) realloc(t2mergedclusters[j].point, t2mergedclusters[j].nelem * sizeof(struct t2));
              *(t2mergedclusters[j].point + t2mergedclusters[j].nelem - 1) = *(t2mergedclusters[i].point + m);
            }
          }
          /* we remove the other one */
          t2mergedclusters[i].nelem = 0;
          FREE(t2mergedclusters[i].point);
        }
      }
    }
  }
  // we now have a proper list of merged clusters
  if (verbose)
    for (j = 0; j < nbt2cluster; j++) {
      stringstream s;
      for (k = 0; k < t2mergedclusters[j].nelem; k++)
        s << (*(t2mergedclusters[j].point + k)).time << " " << (*(t2mergedclusters[j].point + k)).ID << ", ";
      if (t2mergedclusters[j].nelem > 0)
        InternalLog(s.str().c_str(),IKDEBUG);
    }
  /* we build the output */
  for (j = 0; j < nbt2cluster; j++) {
    /* check for doublets to be counted as only one tank */
    tcm = tdn = tmr = 0;
    for (k = 0; k < t2mergedclusters[j].nelem; k++) {
      if ((*(t2mergedclusters[j].point + k)).ID == CARMEN) tcm++;
      if ((*(t2mergedclusters[j].point + k)).ID == MIRANDA) tcm++;
      if ((*(t2mergedclusters[j].point + k)).ID == DIA) tdn++;
      if ((*(t2mergedclusters[j].point + k)).ID == NOCHE) tdn++;
      if ((*(t2mergedclusters[j].point + k)).ID == MOULIN) tmr++;
      if ((*(t2mergedclusters[j].point + k)).ID == ROUGE) tmr++;
    }
    nbstat = t2mergedclusters[j].nelem;
    if (tcm == 2) nbstat--;
    if (tdn == 2) nbstat--;
    if (tmr == 2) nbstat--;
    if (t2mergedclusters[j].nelem > 0 && nbstat >= 3) /* At the end, accept events with 3 stations or more */
      out.push_back(t2mergedclusters[j]);
  }
  // we now have the T3s of this second.
  // adding special T3 algorithms:

  // RANDOM T3
  if (delay != 0)
    delay--;
  else {
    static struct t2list event;
    event.nelem = 1;
    int chosen = (int) (1.0 * nt2 * rand() / (RAND_MAX + 1.0));
    event.point = (struct t2 *) malloc(sizeof(struct t2));
    event.algo = (char *) "RANDOM";
    event.point[0] = input[chosen];
    delay += trivial_algo;
    out.push_back(event);
  }
  // OBSERVER T3
  if (observer)
    for (j = 0; j < nt2; j++)
      if ((input[j].ID == observer)) {
        static struct t2list event;
        event.nelem = 1;
        event.point = (struct t2 *) malloc(sizeof(struct t2));
        event.algo = (char *) "OBSERVER";
        event.point[0] = input[j];
        out.push_back(event);
        observer = 0;
        j = nt2;
      }
  // DOUBLET T3
  if (delaycm != 0) delaycm--;
  if (delaymr != 0) delaymr--;
  if (delaydn != 0) delaydn--;
  for (i = 0; i < nt2cm; i++)
    for (j = 0; j < nt2cm; j++)
      if (((carmenmiranda[i].time - carmenmiranda[j].time) <= dtime
           && (carmenmiranda[i].time - carmenmiranda[j].time) >= -dtime)
          && ((carmenmiranda[i].ID == CARMEN && carmenmiranda[j].ID == MIRANDA && (delaycm == 0))
              || (carmenmiranda[i].ID == MOULIN && carmenmiranda[j].ID == ROUGE && (delaymr == 0))
              || (carmenmiranda[i].ID == DIA && carmenmiranda[j].ID == NOCHE && delaydn == 0))) {
        static struct t2list event;
        if (carmenmiranda[i].ID == CARMEN || carmenmiranda[j].ID == MIRANDA) {
          event.algo = (char *) "CARMEN_MIRANDA";
          delaycm += carmen_miranda;
        }
        if (carmenmiranda[i].ID == DIA || carmenmiranda[j].ID == NOCHE) {
          event.algo = (char *) "DIA_NOCHE";
          delaydn += carmen_miranda;
        }
        if (carmenmiranda[i].ID == MOULIN || carmenmiranda[j].ID == ROUGE) {
          event.algo = (char *) "MOULIN_ROUGE";
          delaymr += carmen_miranda;
        }
        event.nelem = 2;
        event.point = (struct t2 *) malloc(2 * sizeof(struct t2));
        event.point[0] = carmenmiranda[j];
        event.point[1] = carmenmiranda[i];
        out.push_back(event);
      }
  // Finished...
  if (verbose && out.size() != 0) {
    stringstream s;
    s << "Cluster Rate for second " << second_ << " is " << out.size();
    InternalLog(s.str().c_str(),IKINFO);
  }
  if (debug) {
    stringstream s;
    s << "binsize (" << binsize << ") and binnumber (" << binnumber << ") used";
    InternalLog(s.str().c_str(),IKDEBUG);
  }
  return out;
}

/// frees the malloced part of the list of t2
void XbAlgoFree(t2list in) {
  FREE(in.point);
}

/// builds t3 from a list of t2
t3 XbAlgoT3Build(struct t2list in) {
  t3stat tmp;
  struct t3 ret;
  int medusec = 0;

  ret.algo = in.algo;
  ret.refSecond = second_;
  ret.refuSecond = 2000000;
  for (unsigned int i = 0; i < in.nelem; i++)
    if (in.point[i].time + 125 < ret.refuSecond)
      ret.refuSecond = in.point[i].time + 125;
  for (unsigned int i = 0; i < in.nelem; i++) {
    tmp.window = 0;
    tmp.ID = in.point[i].ID;
    tmp.offset = in.point[i].time - ret.refuSecond;
    ret.point.push_back(tmp);
    medusec += tmp.offset;
  }
  // adding extra stations if trigger is not a doublet/observer
  // going up to X crowns, see below
  if (!  (strcmp(in.algo, "MOULIN_ROUGE") == 0
          || strcmp(in.algo, "DIA_NOCHE") == 0
          || strcmp(in.algo, "OBSERVER") == 0
          || strcmp(in.algo, "CARMEN_MIRANDA") == 0))
    for (int i = 0; i < nstat; i++) {
      unsigned int j = 0;
      tmp.ID = array[i].Id;
      int mindist=99;
      while (j < in.nelem && in.point[j].ID != tmp.ID) {
        if (tmp.ID<ROW)
          if (distcrown[tmp.ID*ROW+in.point[j].ID]<mindist)
            mindist=distcrown[tmp.ID*ROW+in.point[j].ID];
        j++;
      }
      if (j == in.nelem && /*tmp.ID < 2048 &&*/ mindist<6) { // 5 crowns, XB 18/07/2013 // in simulations we have stations with id's > 2048
        tmp.window = 30;
        tmp.offset = medusec / ((int)in.nelem);
        ret.point.push_back(tmp);
      }
    }
  if (ret.refuSecond < 0) {
    ret.refSecond--;
    ret.refuSecond += 1000000;
  }
  if (ret.refuSecond > 1000000) {
    ret.refSecond++;
    ret.refuSecond -= 1000000;
  }
  return ret;
}
