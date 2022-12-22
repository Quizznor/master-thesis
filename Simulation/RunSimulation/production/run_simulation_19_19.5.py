#!/usr/bin/python3

import os, sys
import subprocess

# LIBRARIES:
# 15.5 - 18.5 log E = /lsdf/auger/corsika/prague/QGSJET-II.04/proton/ -    NO EXTENSION
# 18.5 - 20.2 log E = /lsdf/auger/corsika/napoli/QGSJET-II.04/proton/ - .part EXTENSION

E_DICT = {
          "16_16.5" : ["prague","*","(1)"],
          "16.5_17" : ["prague","*","(1)"],
          "17_17.5" : ["prague","*","(1)"],
          "17.5_18" : ["prague","*","(1)"],
          "18_18.5" : ["prague","*","(1)"],
          "18.5_19" : ["napoli","*.part","(1).part"],
          "19_19.5" : ["napoli","*.part","(1).part"],
          }

E_RANGE = "19_19.5"
ALREADY_PRESENT = 0
NUM_RETHROWS = 1

SRC_DIR=f"/lsdf/auger/corsika/{E_DICT[E_RANGE][0]}/QGSJET-II.04/proton/{E_RANGE}/"
# SRC_DIR=f"/cr/users/filip/Simulation/TestOutput/"
DESTINATION_DIR=f"/cr/tempdata01/filip/QGSJET-II/protons/{E_RANGE}/"
file_list = [file for file in os.listdir(SRC_DIR) if not '.' in file or file.endswith(".part")]
FILE_NAME = file_list[int(sys.argv[1])]
EVENT_NAME = FILE_NAME.replace(".part", "")

# n_showers in lib
#   30000 16_16.5
#   19996 16.5_17
#   12496 18_18.5
#   10000 17_17.5
#    9998 17.5_18
#    7232 18.5_19
#    8646 19_19.5

for j in range(ALREADY_PRESENT, ALREADY_PRESENT + NUM_RETHROWS):
    NAME = f"{EVENT_NAME}_{str(j).zfill(2)}"
    SEED = str(j).zfill(6)

    try:
        if os.path.isfile(f"{DESTINATION_DIR}/{NAME}.csv"):
            raise IndexError

        bash_arguments = file_list[int(sys.argv[1])], SRC_DIR, DESTINATION_DIR, NAME, E_RANGE, SEED, E_DICT[E_RANGE][1], E_DICT[E_RANGE][2]
        subprocess.call([f"/cr/users/filip/Simulation/RunSimulation/run_simulation.sh", *bash_arguments])

    except IndexError:
        pass