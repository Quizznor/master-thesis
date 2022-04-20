#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plots the convex hull of a set of points in 2 dimensions"""

import numpy as np
import matplotlib.pyplot as plt
from pyik.numpyext import getFractionalConvexHull

np.random.seed(1) # fixed seed for reproducible results

x = np.random.normal(scale=1,size=100)
y = np.random.normal(scale=3,size=100)
points = zip(x,y)

plt.figure()
plt.plot(x,y, ".")

contour = np.array(getFractionalConvexHull(points,.5))

plt.plot(contour[:,0],contour[:,1])

plt.show()
