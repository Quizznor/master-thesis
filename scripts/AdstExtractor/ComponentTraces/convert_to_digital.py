import sys, os, math
import numpy as np

def digitize(array, convert):

    if convert:
        for i, station in enumerate(array):
            for j, bin in enumerate(station):
                array[i][j] = math.floor(bin) / 61.75
    else:
        pass

    return np.array(array)

data = np.loadtxt(sys.argv[1])
os.remove(sys.argv[1])

for i, trace in enumerate(data):
    data[i] = digitize(trace, convert = True)

np.savetxt(sys.argv[1], data)
