#!/usr/bin/python3

import sys, os
import subprocess
import time
import numpy as np

src = "/cr/tempdata01/filip/QGSJET-II/LTP/CrossCheck/19_19.5/"
all_adsts = os.listdir(src)

# os.system("source /cr/users/filip/Simulation/.env/auger_env.sh")
# os.system(f"/cr/users/filip/Simulation/AdstReader/AdstReader 3 {this_file}")

for i in range(1000):
    this_file = src + all_adsts[i]
    subprocess.call([f"/cr/users/filip/Simulation/calculateLTP/run_simulation.sh", this_file])