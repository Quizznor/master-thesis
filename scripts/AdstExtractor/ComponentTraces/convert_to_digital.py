import sys, os, math
import numpy as np

def digitize(array, convert):

    if convert:
        for i in range(len(array)):
            array[i] = math.floor(array[i]) / 61.75
            
    else:
        for i in range(len(array)):
            array[i] = math.floor(array[i]) / 61.75

    return array

data = np.loadtxt(sys.argv[1])
os.remove(sys.argv[1])

for i, trace in enumerate(data):
    data[i] = digitize(trace, convert = True)

np.savetxt(sys.argv[1], data)
