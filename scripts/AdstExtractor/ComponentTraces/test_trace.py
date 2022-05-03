#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import sys, os

data = np.loadtxt(sys.argv[1])
print(data.shape)

for i, trace in enumerate(data):

    if np.max(data) in trace:
        plt.plot(range(len(trace)), trace, label = i)
    else:
        plt.plot(range(len(trace)), trace)

plt.legend()
plt.show()