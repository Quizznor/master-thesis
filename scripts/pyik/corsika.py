# -*- coding: utf-8 -*-
"""Utility functions to read CORSIKA longitudinal files and headers of ground particle files."""


class LongitudinalDataProvider:
  """
  Reads a longitudinal file and provides the data content as numpy arrays.

  The class accepts a filename and automatically reads the file. The arrays are
  accessible as data members of the class. Handles files both written with
  and without the SLANT option.

  Authors
  -------
  Hans Dembinski <hans.dembinski@kit.edu>
  """

  def __init__(self, fn):
    import numpy as np
    import math

    self.fn = fn

    lines = file(fn).readlines()

    words = [x for x in lines[0].split() if x]

    n = int(words[3]) - 1
    self.size = n

    if words[4] == "SLANT":
      self.depthType = "slant"
    elif words[4] == "VERTICAL":
      self.depthType = "vertical"
    else:
      raise StandardError("Could not determine depth type from first line of longitudinal file")

    self.depth = np.empty(n)

    self.nPhoton = np.empty(n)
    self.nElectronP = np.empty(n)
    self.nElectronM = np.empty(n)
    self.nMuonP = np.empty(n)
    self.nMuonM = np.empty(n)
    self.nHadron = np.empty(n)
    self.nNuclei = np.empty(n)
    self.nCherenkov = np.empty(n)

    columns = (self.depth, self.nPhoton,
               self.nElectronP, self.nElectronM,
               self.nMuonP, self.nMuonM,
               self.nHadron, None, self.nNuclei,
               self.nCherenkov)

    i = 0
    for line in lines[2:2 + n]:
      for j, x in enumerate(columns):
        if j == 0:
          if line[:6] == " *****":
            # correct for overflow assuming equal steps in slant depth
            dx = x[1] - x[0]
            x[i] = x[i - 1] + dx
          else:
            x[i] = float(line[:6])
        else:
          if x is None:
            continue
          x[i] = float(line[6 + (j - 1) * 12:6 + j * 12])
      i += 1

    # second to last empty bogus for muons and hadrons (Corsika 6.735 with SLANT option) !
    # extrapolate with power law if value is too low
    def extrapol(entry, x, xp, xpp, n, np, npp):
      log = math.log
      logn = (log(np) - log(npp)) / (xp - xpp) * (x - xp) + log(np)
      if n == 0 or (logn - log(n)) > 0.7:  # = factor of two
        return math.exp(logn)
      else:
        return n

    self.nMuonP[n - 1] = extrapol("MU+", self.depth[n - 1], self.depth[n - 2], self.depth[n - 3],
                                  self.nMuonP[n - 1], self.nMuonP[n - 2], self.nMuonP[n - 3])
    self.nMuonM[n - 1] = extrapol("MU-", self.depth[n - 1], self.depth[n - 2], self.depth[n - 3],
                                  self.nMuonM[n - 1], self.nMuonM[n - 2], self.nMuonM[n - 3])
    self.nHadron[n - 1] = extrapol(
      "HADRONS", self.depth[n - 1], self.depth[n - 2], self.depth[n - 3],
      self.nHadron[n - 1], self.nHadron[n - 2], self.nHadron[n - 3])

    self.eLossPhoton = np.empty(n)
    self.eIonLossEm = np.empty(n)
    self.eCutLossEm = np.empty(n)
    self.eIonLossMuon = np.empty(n)
    self.eCutLossMuon = np.empty(n)
    self.eIonLossHadron = np.empty(n)
    self.eCutLossHadron = np.empty(n)
    self.eLossNeutrino = np.empty(n)

    columns = (self.eLossPhoton,
               self.eIonLossEm, self.eCutLossEm,
               self.eIonLossMuon, self.eCutLossMuon,
               self.eIonLossHadron, self.eCutLossHadron,
               self.eLossNeutrino, None)

    i = 0
    for line in lines[5 + n:5 + 2 * n]:
      for j, x in enumerate(columns):
        if x is None:
          continue
        x[i] = float(line[7 + j * 12:7 + (j + 1) * 12])
      i += 1


class SteeringDataProvider:
  """
  Reads a steering card and provides the data content.

  The class accepts a filename and automatically reads the file.
  The data in the steering card is provided in form of attributes of the class.

  Limitations
  -----------
  This class is far from complete. I added only the most interesting things.

  Authors
  -------
  Hans Dembinski <hans.dembinski@kit.edu>
  Detlef Maurel <detlef.maurel@kit.edu>
  """

  def __init__(self, fn):
    lines = file(fn).readlines()

    for line in lines:
      words = [x for x in line.split() if x]
      if not words:
        continue
      key = words[0]
      if key == "RUNNR":
        self.runnr = int(words[1])
      elif key == "PRMPAR":
        self.primary = int(words[1])
      elif key == "ERANGE":
        self.energyRange = float(words[1]) * 1e9, float(words[2]) * 1e9  # in eV
      elif key == "ESLOPE":
        self.energySlope = float(words[1])
      elif key == "THETAP":
        self.thetaRange = float(words[1]), float(words[2])  # in Deg
      elif key == "PHIP":
        self.phiRange = float(words[1]), float(words[2])  # in Deg

      elif key == "THIN":
        self.thinning = [float(words[1]), float(words[2]), float(words[3])]
      elif key == "ATMOD":
        self.atmod = words[2]

      # atmospheric profile, parameters for the top of the atmosphere
      # are added (see CORSIKA manual)
      elif key == "ATMA":
        self.atma = [float(words[1]), float(words[2]), float(words[3]), float(words[4])]
        self.atma.append(0.01128292)
      elif key == "ATMB":
        self.atmb = [float(words[1]), float(words[2]), float(words[3]), float(words[4])]
        self.atmb.append(1.)
      elif key == "ATMC":
        self.atmc = [float(words[1]), float(words[2]), float(words[3]), float(words[4])]
        self.atmc.append(1.e9)
      elif key == "ATMLAY":
        self.atmlay = [float(words[1]), float(words[2]), float(words[3]), float(words[4])]
        self.atmlay.insert(0, 0.)

      # extend here


def IsDataFileValid(filename):
  """
  Test whether a CORSIKA particle file is comlete (has a RUN end tag).

  Authors
  -------
  Hans Dembinski <hans.dembinski@kit.edu>
  """
  import os
  min_size = 26215
  if os.path.exists(filename) and os.stat(filename).st_size > min_size:
    f = file(filename, "rb")
    f.seek(-min_size, 2)
    if 'RUNE' in f.read(min_size):
      return True
  return False


knownBadCorsikaFiles = (
    "/data/corsdat2/joe/e19m100a65/DAT129625",
    "/data/corsdat2/joe/e19m100a65/DAT129648",
    "/data/FeQGSJet/DAT010657.part",
    "/data/FeQGSJet/DAT010768.part",
    "/data/FeQGSJet/DAT011059.part",
    "/data/FeQGSJet/DAT011123.part",
    "/data/FeQGSJet/DAT011591.part",
    "/data/FeQGSJet/DAT011626.part",
    "/data/FeQGSJet/DAT012303.part",
    "/data/FeQGSJet/DAT012475.part",
    "/data/FeQGSJet/DAT013042.part",
    "/data/FeQGSJet/DAT013250.part",
    "/data/FeQGSJet/DAT013325.part",
    "/data/FeQGSJet/DAT013343.part",
    "/data/FeQGSJet/DAT013354.part",
    "/data/pQGSJet/DAT000989.part",
    "/data/pQGSJet/DAT001036.part",
    "/data/pQGSJet/DAT001138.part",
    "/data/pQGSJet/DAT002327.part",
    "/data/pQGSJet/DAT002391.part"
)


def IsKnownBadFile(path):
  """Test whether a CORSIKA particle file is in the aforementioned list of known bad files."""
  return path in knownBadCorsikaFiles


def scanGroundParticleFile(filename):
  """
  Read a CORSIKA particle and returns a dictionary that contains the properties of the shower.

  Parameters
  ----------
  filename: name of CORSIKA ground particle file

  Examples
  --------
  >>> from pyik import corsika
  >>> corsika.scanGroundParticleFile("/data/pQGSJet/DAT000841.part")
  {'id': 841, 'energy': 10000000000.0, 'primary': 2212, 'zenith': 0.0, 'azimuth': 0.0}

  Limitations
  -----------
  Only one shower per file is supported (can be extended).

  Authors
  -------
  Detlef Maurel <detlef.maurel@kit.edu>
  """
  from corsikainterface import CorsikaShowerFile

  f = CorsikaShowerFile(filename)
  f.Read()

  info = {}
  info["primary"] = f.GetCurrentShower().GetPrimary()
  info["energy"] = f.GetCurrentShower().GetEnergy()
  info["zenith"] = f.GetCurrentShower().GetZenith()
  info["azimuth"] = f.GetCurrentShower().GetAzimuth()
  info["id"] = f.GetCurrentShower().GetShowerRunId()

  return info


def createShowerInfoLibrary(corsikafilenames, libraryfilename):
  """
  Read many CORSIKA particle files and creates a library that contains the properties of the shower.

  The library is saved to a shelve file.

  Parameters
  ----------
  corsikafilenames: list of CORSIKA ground particle file names
  libraryfilename: name of shelve file

  Examples
  --------
  >>> from pyik import corsika
  >>> from glob import glob
  >>> corsika.createShowerInfoLibrary( glob("/data/pQGSJet/DAT00084*part"), "mylibrary.shelve")
  Scanning file 10/10: /data/pQGSJet/DAT000849.part

  Authors
  -------
  Detlef Maurel <detlef.maurel@kit.edu>
  """
  from sys import stdout
  from os import path
  import shelve

  library = []
  for ifn, fn in enumerate(sorted(corsikafilenames)):
    if not path.exists(fn):
      print "### warning: file not found", fn
      continue
    if fn in knownBadCorsikaFiles:
      print fn, "marked as bad, not reading it."
      continue
    if not IsDataFileValid(fn):
      print fn, "has no valid run end tag, not reading it."
      continue
    stdout.write("Scanning file %i/%i: %s    \r" % (ifn + 1, len(corsikafilenames), fn))
    stdout.flush()
    showerInfo = scanGroundParticleFile(fn)
    showerInfo["filename"] = fn

    # steering data info
    fn_lst = "{}.lst".format(fn)
    if path.exists(fn_lst):
      lst = SteeringDataProvider(fn_lst)
      try:
        showerInfo["runnr"] = lst.runnr
      except:
        print "RunNr not available in steering file of {}!".format(fn)
      try:
        showerInfo["atmosphere"] = lst.atmod
      except:
        print "Atmosphere information not present in steering file of {}!".format(fn)
    else:
      print "Cannot read steering file, information will not be available!"

    library.append(showerInfo)

  stdout.write("\n\n")
  shelve.open(libraryfilename)["library"] = library
