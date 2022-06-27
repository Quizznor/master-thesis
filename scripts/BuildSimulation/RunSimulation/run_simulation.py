#!/usr/bin/python3

import os, sys
import subprocess

# Check source file extension first and make sure to match it in ./bootstrap_xml.in and edit line 10

# LIBRARIES:
# 15.5 - 18.5 log E = /lsdf/auger/corsika/prague/QGSJET-II.04/proton/ - NO EXTENSION
# 18.5 - 20.2 log E = /lsdf/auger/corsika/napoli/QGSJET-II.04/proton/ - .part EXTENSION

# don't forget to change directories in awk!!!!
E_RANGE = "19_19.5"

SRC_DIR=f"/lsdf/auger/corsika/napoli/QGSJET-II.04/proton/{E_RANGE}/"
DESTINATION_DIR=f"/cr/tempdata01/filip/QGSJET-II/protons/{E_RANGE}/"
file_list = [file for file in os.listdir(SRC_DIR) if not '.' in file or file.endswith(".part")]
FILE_NAME = file_list[int(sys.argv[1])]
EVENT_NAME = FILE_NAME.replace(".part", "")

if not os.path.isfile(f"{DESTINATION_DIR}/{file_list[int(sys.argv[1])]}.csv"):
    subprocess.call([f"/cr/users/filip/scripts/BuildSimulation/RunSimulation/run_simulation.sh", file_list[int(sys.argv[1])], SRC_DIR, DESTINATION_DIR, EVENT_NAME])