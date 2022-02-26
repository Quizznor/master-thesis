#!/usr/bin/python3

import os, sys
import numpy

# think of something more efficient/smarter here when working with larger data samples
data_directory = "/lsdf/auger/tmp/hahn/test_production/napoli/ub/QGSJET-II.04/proton/19_19.5/00/"
simulation_data = os.listdir(data_directory)

# trigger_disabled = [file for file in simulation_data if "trigger" in file]
# trigger_enabled = [file for file in simulation_data if "trigger" not in file]

os.system(f"/cr/users/filip/scripts/AdstExtractor/AdstExtractor {''.join([data_directory, simulation_data[int(sys.argv[1])]])}")