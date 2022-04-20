# -*- coding: utf-8 -*-

"""
PyIK - The Python Instrument Kit
--------------------------------
This package contains a set of tools that extend and simplify common
analysis tasks. The future aim of this package is nothing less than
to provide a working alternative to ROOT.

The tools are grouped by topic into several modules which are listed below.
Most of them depend on external modules which are not shipped with Python.
The respective dependencies are also listed.

Content
-------
adst           : Contains tools to work with ADST files
  Requires an installation of ADST or Offline

corsika        : Contains tools to work with CORSIKA files

database       : Contains a hierarchical database
  Requires SQL alchemy

fit            : Contains classes and functions for function minimization
  Requires nlopt

locked_shelve  : Functionality to read shelve files and prevent write collisions

mplext         : Extends matplotlib
  Requires matplotlib

numpyext       : Extends numpy
  Requires numpy

performance    : Contains tools to increase performance (e.g. easy parallelization)
  Requires progressbar

record_figure  : Contains tools to record matplotlib figures or other objects

rext           : Contains some interface functions to the programming language R
  Requires rpy2 interface

rootext        : Functions to convert python to ROOT objects and vice-versa

time_conversion: Contains tools to convert between UTC and GPS

Notes
-----
This packages also contains some a directory with working examples
to copy-paste from.

Maintainers
-----------
Hans Dembinski <hans.dembinski@kit.edu>
Benjamin Fuchs <benjamin.fuchs@kit.edu>
Felix Werner <Felix.Werner@kit.edu>
Alexander Schulz <alexander.schulz@kit.edu>
Ariel Bridgeman <ariel.bridgeman@kit.edu>
"""
