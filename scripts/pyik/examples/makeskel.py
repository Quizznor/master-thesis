#!/usr/bin/env python
# -*- coding: utf-8 -*-
__doc__ = """Usage: makeskel.py <file>
Creates a skeleton for <file>. The type of the skeleton is selected through the extension.
Supported: python, shell
"""

# skeletons
pythonScript = """#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import numpy as np
from matplotlib import pyplot as plt
from math import *
from pyik.adst import RecEventProvider
from pyik.performance import *

"""

shell = """#!/bin/sh

"""

rootScript = """#if !defined(__CINT__) || defined(__MAKECINT__)
  #include <TFile.h>
  #include <TTree.h>
  #include <TMath.h>
  #include <TLegend.h>
  #include <TH1D.h>
  #include <TH2D.h>

  #include <iostream>
  #include <cstdlib>
  #include <cmath>

  #include <Styler.h>

  using namespace std;
#endif

void %s() {
  TFile* f = TFile::Open("myfile.root","READ");
}
"""

cppfile = """#include "%s.h"

#include <iostream>
#include <vector>

using namespace std;




"""

cppheader = """#ifndef %s_h
#define %s_h

#include <iostream>
#include <cmath>
#include <vector>





#endif

"""


# main
from sys import argv
from os import chmod, path, system


def push(txt, filename):
  file(filename, "w").write(txt)

if len(argv) < 2:
  print __doc__
  raise SystemExit

filename = argv[1]

if path.exists(filename):
  print "Warning! %s already exists. I won't override it." % filename
  raise SystemExit

fn, ext = path.splitext(filename)

if ext == ".py":
  push(pythonScript, filename)
  chmod(filename, 0755)
elif ext == ".sh":
  push(shell, filename)
  chmod(filename, 0755)
elif ext == ".C":
  push(rootScript % fn, filename)
elif ext == ".cpp" or ext == ".cc":
  push(cppfile % (fn), filename)
elif ext == ".h" or ext == ".hpp" or ext == ".hh" or ext == ".H":
  push(cppheader % (fn, fn), filename)
else:
  print "Could not create skeleton for %s\nExtension not recognized" % filename
  raise SystemExit
