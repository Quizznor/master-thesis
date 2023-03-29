#!/usr/bin/python3

import os, sys
import subprocess

root_path = "/cr/tempdata01/filip/QGSJET-II/LTP/19_19.5/"
files = [root_path + file for file in os.listdir(root_path)]

subprocess.call([f"/cr/users/filip/Trigger/OfflineComparison/make_CSVs/submit.sh", files[int(sys.argv[1])]])