#!/usr/bin/python3

import os, sys
import subprocess

# Check source file extension first and make sure to match it in ./bootstrap_xml.in and edit line 10

SRC_DIR="/lsdf/auger/corsika/prague/QGSJET-II.04/proton/16.5_17/"
DESTINATION_DIR="/cr/tempdata01/filip/protons/16.5_17"
file_list = [file for file in os.listdir(SRC_DIR) if not file.endswith(".long")]

if not os.path.isfile(f"{DESTINATION_DIR}/{file_list[int(sys.argv[1])]}.csv"):
    print(int(sys.argv[1]), f"{file_list[int(sys.argv[1])]} must be created")
    subprocess.call([f"/cr/data01/filip/SdRecEvent/run_simulation.sh", file_list[int(sys.argv[1])], SRC_DIR])