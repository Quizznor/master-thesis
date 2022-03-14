#!/user/bin/python

import sys, os
import numpy as np

heatmap = np.zeros((600, 2048))
np.savetxt(f"/cr/users/filip/data/second_simulation/heatmap/{sys.argv[1]}.csv", heatmap)

with open(f"/cr/users/filip/data/second_simulation/heatmap/{sys.argv[1]}_binaries.txt","w") as file:
    file.write("0\n0")