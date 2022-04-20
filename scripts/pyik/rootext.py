# -*- coding: utf-8 -*-
import types
import os
import sys
import numpy as np
import tempfile
import ROOT

__all__ = ["ROOT", "ToVector", "ToNumpy", "PyTFile"]


class PyTFile(ROOT.TFile):

  def __init__(self, *args):
    self.save = ROOT.gDirectory
    ROOT.TFile.__init__(self, *args)
    ROOT.gDirectory = self.save

  def __enter__(self):
    return self

  def __exit__(self, type, value, traceback):
    self.Close()

with tempfile.NamedTemporaryFile() as f:
  f.write("""#include <vector>

#define GetVectorBuffer(T,x) \
  T* GetVectorBuffer##x(std::vector<T>& v) { return &v[0]; }

GetVectorBuffer(double, D)
GetVectorBuffer(float, F)
GetVectorBuffer(long, L)
GetVectorBuffer(unsigned long, UL)
GetVectorBuffer(int, I)
GetVectorBuffer(unsigned int, UI)
GetVectorBuffer(short, S)
GetVectorBuffer(unsigned short, US)
""")
  f.flush()
  ROOT.gROOT.LoadMacro(f.name)
  f.close()
  del f


class NDArray(np.ndarray):

  def __new__(subtype, shape, dtype, buffer, parent, offset=0,
              strides=None, order=None, info=None):
    obj = np.ndarray.__new__(subtype, shape, dtype, buffer, offset, strides, order)
    obj.parent = parent
    return obj

  def __array_finalize__(self, obj):
    if obj is None:
      return
    self.parent = getattr(obj, "parent", None)


def ToVector(array):
  """
  Turns a Python list or Numpy array into a ROOT std::vector instance by copying the data.

  Examples
  --------
  >>> a = (1, 2, 3)
  >>> ra = ToVector(a)
  >>> print type(ra), ra[0], ra[1], ra[2]
  <class 'ROOT.vector<int,allocator<int> >'> 1 2 3

  Authors
  -------
  Hans Dembinski <hans.dembinski@kit.edu>
  """

  n = len(array)
  if type(array[0]) == types.StringType:
    res = ROOT.std.vector("string")(n)
  elif type(array[0]) == types.FloatType or (type(array) == np.ndarray and np.issubdtype(array[0].dtype, np.float)):
    res = ROOT.std.vector("double")(n)
  elif type(array[0]) == types.IntType or (type(array) == np.ndarray and np.issubdtype(array[0].dtype, np.float)):
    res = ROOT.std.vector("int")(n)
  for i in xrange(n):
    res[i] = array[i]
  return res


def ToNumpy(x):
  """
  Turns ROOT data structures into numpy structures.

  Fast treatment (no copy):
  std::vector<T>

  All other objects are copied.

  Examples
  --------
  >>> a = ROOT.std.vector("double")(3)
  >>> a[0] = 1; a[1] = 2; a[2] = 3
  >>> na = ToNumpy(a)
  >>> print type(na), na
  <type 'numpy.ndarray'> [ 1.  2.  3.]

  Authors
  -------
  Hans Dembinski <hans.dembinski@kit.edu>
  """
  if isinstance(x, str):
    f = ROOT.TFile.Open(x)
    d = {}
    for key in f.GetListOfKeys():
      d[key.GetName()] = ToNumpy(key.ReadObj())
    f.Close()
    return d
  elif isinstance(x, ROOT.TVectorD):
    return NDArray((x.GetNoElements(),), np.float64, x.GetMatrixArray(), x)
  elif isinstance(x, ROOT.TMatrixD) or isinstance(x, ROOT.TMatrixDSym):
    # return NDArray((x.GetNrows(),x.GetNcols()), np.float64, x.GetMatrixArray(), x)
    a = np.empty((x.GetNrows(), x.GetNcols()), np.float64)
    for ix in xrange(x.GetNrows()):
      for iy in xrange(x.GetNcols()):
        a[ix, iy] = x(ix, iy)
    return a
  # here a reference to the original object is stored so that
  # the original object is not deleted by gargabe collection
  elif isinstance(x, ROOT.std.vector("double")):
    return NDArray((x.size(),), np.float64, ROOT.GetVectorBufferD(x), x)
  elif isinstance(x, ROOT.std.vector("float")):
    return NDArray((x.size(),), np.float32, ROOT.GetVectorBufferF(x), x)
  elif isinstance(x, ROOT.std.vector("long")):
    return NDArray((x.size(),), np.int64, ROOT.GetVectorBufferL(x), x)
  elif isinstance(x, ROOT.std.vector("unsigned long")):
    return NDArray((x.size(),), np.uint64, ROOT.GetVectorBufferUL(x), x)
  elif isinstance(x, ROOT.std.vector("int")):
    return NDArray((x.size(),), np.int32, ROOT.GetVectorBufferI(x), x)
  elif isinstance(x, ROOT.std.vector("unsigned int")):
    return NDArray((x.size(),), np.uint32, ROOT.GetVectorBufferUI(x), x)
  elif isinstance(x, ROOT.std.vector("short")):
    return NDArray((x.size(),), np.int16, ROOT.GetVectorBufferS(x), x)
  elif isinstance(x, ROOT.std.vector("unsigned short")):
    return NDArray((x.size(),), np.uint16, ROOT.GetVectorBufferUS(x), x)
  elif isinstance(x, ROOT.TH3):
    nx = x.GetXaxis().GetNbins()
    ny = x.GetYaxis().GetNbins()
    nz = x.GetZaxis().GetNbins()
    xedges = np.empty(nx + 1)
    yedges = np.empty(ny + 1)
    zedges = np.empty(nz + 1)
    h = np.empty((nx, ny, nz))
    for ix in xrange(nx):
      xedges[ix] = x.GetXaxis().GetBinLowEdge(ix + 1)
    xedges[nx] = x.GetXaxis().GetBinLowEdge(nx + 1)
    for iy in xrange(ny):
      yedges[iy] = x.GetYaxis().GetBinLowEdge(iy + 1)
    yedges[ny] = x.GetYaxis().GetBinLowEdge(ny + 1)
    for iz in xrange(nz):
      zedges[iz] = x.GetZaxis().GetBinLowEdge(iz + 1)
    zedges[nz] = x.GetZaxis().GetBinLowEdge(nz + 1)
    for ix in xrange(nx):
      for iy in xrange(ny):
        for iz in xrange(nz):
          h[ix, iy, iz] = x.GetBinContent(ix + 1, iy + 1, iz + 1)
    return h, xedges, yedges, zedges
  elif isinstance(x, ROOT.TH2):
    nx = x.GetXaxis().GetNbins()
    ny = x.GetYaxis().GetNbins()
    xedges = np.empty(nx + 1)
    yedges = np.empty(ny + 1)
    h = np.empty((nx, ny))
    for ix in xrange(nx):
      xedges[ix] = x.GetXaxis().GetBinLowEdge(ix + 1)
    xedges[nx] = x.GetXaxis().GetBinLowEdge(nx + 1)
    for iy in xrange(ny):
      yedges[iy] = x.GetYaxis().GetBinLowEdge(iy + 1)
    yedges[ny] = x.GetYaxis().GetBinLowEdge(ny + 1)
    for ix in xrange(nx):
      for iy in xrange(ny):
        h[ix, iy] = x.GetBinContent(ix + 1, iy + 1)
    return h, xedges, yedges
  elif isinstance(x, ROOT.TProfile):
    n = x.GetXaxis().GetNbins()
    xedges = np.empty(n + 1)
    hs = np.empty(n)
    hes = np.empty(n)
    for i in xrange(n):
      xedges[i] = x.GetXaxis().GetBinLowEdge(i + 1)
      hs[i] = x.GetBinContent(i + 1)
      hes[i] = x.GetBinError(i + 1)
    xedges[n] = x.GetXaxis().GetBinLowEdge(n + 1)
    return hs, hes, xedges
  elif isinstance(x, ROOT.TH1):
    n = x.GetXaxis().GetNbins()
    xedges = np.empty(n + 1)
    h = np.empty(n)
    for i in xrange(n):
      xedges[i] = x.GetXaxis().GetBinLowEdge(i + 1)
      h[i] = x.GetBinContent(i + 1)
    xedges[n] = x.GetXaxis().GetBinLowEdge(n + 1)
    return h, xedges
  elif isinstance(x, ROOT.TGraphErrors):
    n = x.GetN()
    xs = np.ndarray((n,), dtype=np.float64, buffer=x.GetX())
    ys = np.ndarray((n,), dtype=np.float64, buffer=x.GetY())
    xes = np.ndarray((n,), dtype=np.float64, buffer=x.GetEX())
    yes = np.ndarray((n,), dtype=np.float64, buffer=x.GetEY())
    return xs, ys, xes, yes
  elif isinstance(x, ROOT.TGraphAsymmErrors):
    n = x.GetN()
    xs = np.ndarray((n,), dtype=np.float64, buffer=x.GetX())
    ys = np.ndarray((n,), dtype=np.float64, buffer=x.GetY())
    xeds = np.ndarray((n,), dtype=np.float64, buffer=x.GetEXlow())
    xeus = np.ndarray((n,), dtype=np.float64, buffer=x.GetEXhigh())
    yeds = np.ndarray((n,), dtype=np.float64, buffer=x.GetEYlow())
    yeus = np.ndarray((n,), dtype=np.float64, buffer=x.GetEYhigh())
    return xs, ys, (xeds, xeus), (yeds, yeus)
  elif isinstance(x, ROOT.TVector2):
    return np.asfarray([x.X(), x.Y()])
  elif isinstance(x, ROOT.TVector3):
    return np.asfarray([x.X(), x.Y(), x.Z()])
  else:
    raise ValueError("Cannot handle type %s yet, please\
                      have a look at pyik.rootext and implement it" % (type(x)))

if __name__ == "__main__":
  import doctest
  doctest.testmod()
