#!/usr/bin/python3

import os, sys

source = "/cr/data/quentin/Comparison_UB_UUB/output-cont-UUB/napoli/prot-epos/lge19.0-19.5/01/"
ADSTs = [os.path.join(source, file) for file in os.listdir(source)]

os.system(f"/cr/users/filip/Simulation/LTP_cross_check/submit.sh {ADSTs[int(sys.argv[1])]}")

