# -*- coding: utf-8 -*-
"""Functions to work with ADSTs in python."""
import ROOT
import os

FULL = 0  # read everything
MICRO = 1  # only shower level observables
MINI = 2  # only traces are turned off


def __LocatedLibraries():
  for base in ("$AUGEROFFLINEROOT", "$ADSTROOT"):
    xbase = os.path.expandvars(base)
    if xbase and os.path.exists(xbase):
      for p in [xbase + "/lib/" + x for x in os.listdir(xbase + "/lib/")]:
        if "receventkg" in p.lower():
          yield p

for path in __LocatedLibraries():
  ROOT.gSystem.Load(path)
  break
else:
  raise EnvironmentError("Could not find RecEventKG library")


def RecEventProvider(inp, mode=FULL):
  """
  Provide an iterator over RecEvent objects saved in one or several ADST files.

  Parameters
  ----------
  inp: string or list
    This can be: A path to a ROOT file, a path to a text file, which contains one
    path of a ROOT file per line, a list of paths to several ROOT files.
  mode: FULL, MICRO, MINI or list of strings
    FULL = read everything (default)
    MICRO = read only high-level observables (station info is turned off!)
    MINI = read everything except traces
    If this is a list of strings it is interpreted as a list of branches to be read out,
    while everything else is ignored (for the power user).

  Examples
  --------
  >>> for event in RecEventProvider("example.root"):
  ...   print event.GetSDEvent().GetEventId()

  Authors
  -------
  Hans Dembinski <hans.dembinski@kit.edu>
  """
  import types

  # BEWARE: We don't use SmartOpen, it has a memory leak.

  filenames = []
  if type(inp) == types.StringType:
    # is a string
    if inp[-5:] == ".root":
      # is name of a single root file
      filenames.append(inp)
    else:
      # is name of file with list of root files
      for line in file(inp):
        if line[0] == "#":
          continue
        line = line.strip()
        if not line:
          continue
        filenames.append(line)
  elif hasattr(inp, "__len__"):
    # is a collection
    filenames = inp
  else:
    # is something else
    raise TypeError("inp has to be string of a collection of strings")

  if len(filenames) == 0:
    raise StandardError("list of filenames is empty")

  filenames.sort()

  vfilenames = ROOT.std.vector("string")()
  for filename in filenames:
    vfilenames.push_back(filename)
  del filenames

  rf = ROOT.RecEventFile(vfilenames)
  if hasattr(mode, "__iter__"):
    rf.SetBranchStatus("event.*", 0)
    for x in mode:
      rf.SetBranchStatus(x, 1)
  elif mode == MICRO:
    rf.SetMicroADST()
  elif mode == MINI:
    rf.SetMiniADST()

  rev = ROOT.RecEvent()
  rf.SetBuffers(rev)

  while rf.ReadNextEvent() == ROOT.RecEventFile.eSuccess:
    yield rev

  rf.Close()
  del rf
  del vfilenames
  del rev


def GetDetectorGeometry(inp):
  """
  Return the latest DetectorGeometry object from one or several ADST files.

  Parameters
  ----------
  inp: string
    Either path of a ROOT file, path of a text file, which contains one
    path of a ROOT file per line, or list of paths to ROOT files.

  Notes
  -----
  If you work on several ADST files, some might have an older DetectorGeometry
  stored than others, where not all SD stations/FD eyes where available yet.
  This function returns the latest DetectorGeometry object in this case,
  whereas "latest" means the one with most SD stations.

  Examples
  --------
  >>> detgeom = GetDetectorGeometry("example.root")

  Authors
  -------
  Hans Dembinski <hans.dembinski@kit.edu>
  """
  if inp[-5:] == ".root":
    geometry = ROOT.DetectorGeometry()
    rf = ROOT.RecEventFile(inp)
    rf.ReadDetectorGeometry(geometry)
    return geometry

  else:
    # BEWARE: Don't use SmartOpen! It has a memory leak! This is a work-around!

    if hasattr(inp, "__len__"):
      fns = inp
    else:
      try:
        fns = [f for f in file(inp) if f[0] != "#"]
      except:
        raise IOError("Expecting {0} to be either a path to a ROOT file, \
                        a list of paths or path to a file containing a list of paths!".format(inputFileName))

    best = ROOT.DetectorGeometry()
    for line in fns:
      if line[0] == "#":
        continue
      line = line.strip()
      if not line:
        continue
      rf = ROOT.RecEventFile(line)
      geometry = ROOT.DetectorGeometry()
      rf.ReadDetectorGeometry(geometry)
      if geometry.GetNStations() > best.GetNStations():
        best = geometry
    return best


def GetFileInfo(inp):
  """
  Return the FileInfo-object of an ADST file.

  This can be usefull to access the OffLine Configuration
  used to create an ADST file. (see example)

  Parameters
  ----------
  inp: string
    Path of a ROOT file.

  Examples
  --------
  >>> fileinfo = GetFileInfo("example.root")
  >>> fileinfo.GetOfflineConfiguration()

  Authors
  -------
  Benjamin Fuchs <Benjamin.Fuchs@kit.edu>
  """
  if inp[-5:] == ".root":
    fileinfo = ROOT.FileInfo()
    rf = ROOT.RecEventFile(inp)
    rf.ReadFileInfo(fileinfo)
    return fileinfo
