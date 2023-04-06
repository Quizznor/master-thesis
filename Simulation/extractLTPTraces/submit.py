#!/usr/bin/python3

import os, sys
import subprocess

root_path = "/cr/tempdata01/filip/QGSJET-II/LTP/19_19.5/"
all_files = [root_path + file for file in os.listdir(root_path)]

subprocess.call([f"/cr/users/filip/Simulation/extractLTPTraces/submit.sh", all_files[int(sys.argv[1])]])