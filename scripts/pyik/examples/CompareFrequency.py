#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compares the selection efficiency in two data sets as a function of zenith angle, candidate stations and energy (for very inclined events)."""

import numpy as np
from math import *
import cPickle
import os

cacheFilename = "CacheCompareFrequency.pkl"

if not os.path.exists(cacheFilename):
  import sys
  from pyik.adst import RecEventProvider

  input1 = sys.argv[1]
  input2 = sys.argv[2]

  def fill(rs, d):
    sde = rev.GetSDEvent()
    rs = sde.GetSdRecShower()

    a = np.empty(4)

    a[0] = rs.GetS1000()
    a[1] = rs.GetZenith()
    n = 0
    for i in xrange(1000):
      st = sde.GetStation(i)
      if not st:
        break
      if st.GetTotalSignal() > 3.0:
        n += 1
    a[2] = n
    a[3] = sde.GetT5Trigger() & 2

    d[sde.GetEventId()] = a

  d1 = {}
  for rev in RecEventProvider(input1):
    fill(rev, d1)
    # if len(d1)>100:break

  d2 = {}
  for rev in RecEventProvider(input2):
    fill(rev, d2)
    # if len(d2)>100:break

  common = set()
  for key in d1:
    common.add(key)
  for key in d2:
    common.add(key)

  def mean(a, b):
    return 0.5 * (a + b)

  d3 = np.empty((5, len(common)))
  i = 0
  for key in common:
    if key in d1 and key in d2:
      a1 = d1[key]
      a2 = d2[key]
      d3[0][i] = 0
      for ii in xrange(1, 5):
        d3[ii][i] = mean(a1[ii - 1], a2[ii - 1])
    elif key in d1:
      a1 = d1[key]
      d3[0][i] = -1
      for ii in xrange(1, 5):
        d3[ii][i] = a1[ii - 1]
    else:
      a2 = d2[key]
      d3[0][i] = 1
      for ii in xrange(1, 5):
        d3[ii][i] = a2[ii - 1]
    i += 1

  cPickle.dump(d3, file(cacheFilename, "wb"), -1)


# analysis
from matplotlib import pyplot as plt


def Sin2(x):
  return np.square(np.sin(x))


def InvSin2(x):
  return np.arcsin(np.sqrt(x))


def LgEnergy(x):
  return 1.051 * np.log10(x) + np.log10(4.723e18)

from matplotlib.colors import LinearSegmentedColormap
coneho = LinearSegmentedColormap.from_list(
  "coneho", ((0.0, 0.0, 1.0), (1.0, 1.0, 1.0), (1.0, 0.0, 0.0)))


def Plot(fig, h1, h2, xedge, yedge, semilogx, xlabel, ylabel):
  hx1 = np.sum(h1, 1)
  hx2 = np.sum(h2, 1)
  hy1 = np.sum(h1, 0)
  hy2 = np.sum(h2, 0)

  from mpl_toolkits.axes_grid import make_axes_locatable

  axMain = plt.subplot(111)
  divider = make_axes_locatable(axMain)
  axHistx = divider.new_vertical(1.2, pad=0.7, sharex=axMain)
  axHisty = divider.new_horizontal(1.2, pad=0.7, sharey=axMain)
  fig.add_axes(axHistx)
  fig.add_axes(axHisty)

  zlabel = r"$\frac{N_\mathrm{new}-N_\mathrm{old}}{\langle N \rangle}$"

  X, Y = np.meshgrid(xedge, yedge)
  pc = axMain.pcolor(X, Y, np.ma.masked_invalid(2 * h1 / h2).T, vmin=-1.0, vmax=1.0, cmap=coneho)
  axMain.set_xlabel(xlabel)
  axMain.set_ylabel(ylabel)
  if semilogx:
    axMain.semilogx()
  axMain.set_ylim(60, 82)

  from pyik.numpyext import centers

  vmax = 0.5
  vmin = -vmax

  if semilogx:
    axHistx.semilogx()
  axHistx.plot(centers(xedge)[0], np.ma.masked_invalid(2 * hx1 / hx2))
  axHistx.set_ylim(vmin, vmax)
  for x in axHistx.get_xticklabels():
    x.set_visible(False)
  axHistx.set_yticks(np.linspace(vmin, vmax, 3))
  axHistx.set_ylabel(zlabel)
  axHistx.axhline(c="gray", linestyle="dashed", zorder=-1)

  axHisty.plot(np.ma.masked_invalid(2 * hy1 / hy2), centers(yedge)[0])
  axHisty.set_xticks(np.linspace(vmin, vmax, 3))
  axHisty.set_xlabel(zlabel)
  axHisty.set_xlim(vmin, vmax)
  for x in axHisty.get_yticklabels():
    x.set_visible(False)
  axHisty.set_ylim(60, 82)
  axHisty.axvline(c="gray", linestyle="dashed", zorder=-1)

  fig.colorbar(pc, ax=axMain, orientation="horizontal", fraction=0.05, pad=0.2).set_label(zlabel)

d = cPickle.load(file(cacheFilename, "rb"))

if 1:  # E,theta plot

  # filter: 0 = selection, 1 = rmu, 2 = theta, 3 = ncand, 4 = is T5
  def myfilter(x):
    return x[3] > 3
  dd = np.transpose(filter(myfilter, d.T))

  mybins = (30, 20)
  myrange = ((17, 20),
             Sin2(np.radians(np.array((60., 82)))))

  h1, xedge, yedge = np.histogram2d(LgEnergy(dd[1]), Sin2(
    dd[2]), bins=mybins, range=myrange, weights=dd[0])
  h2 = np.histogram2d(LgEnergy(dd[1]), Sin2(dd[2]), bins=mybins, range=myrange)[0]

  fig = plt.figure(1, (9, 10))
  Plot(fig, h1, h2, 10**xedge, np.degrees(InvSin2(yedge)), 1, r"$E /$eV", r"$\theta/^\circ$")
  fig.savefig("FrequencyAtLeast4Cand.pdf")

if 1:  # ncand,theta plot

  fig = plt.figure(2, (9, 10))

  mybins = (25, 20)
  myrange = ((3, 28),
             Sin2(np.radians(np.array((60., 82)))))

  h1, xedge, yedge = np.histogram2d(d[3], Sin2(d[2]), bins=mybins, range=myrange, weights=d[0])
  h2 = np.histogram2d(d[3], Sin2(d[2]), bins=mybins, range=myrange)[0]

  Plot(fig, h1, h2, xedge, np.degrees(InvSin2(yedge)),
       0, r"$N_\mathrm{candidate}$", r"$\theta/^\circ$")
  fig.savefig("Frequency.pdf")

plt.show()
