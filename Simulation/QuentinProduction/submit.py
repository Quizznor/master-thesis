#!/usr/bin/python3

import os, sys

energy = "lge18.0-18.5"

source = f"/cr/data/quentin/Comparison_UB_UUB/output-cont-UUB/napoli/prot-epos/{energy}/01/"
ADSTs = [os.path.join(source, file) for file in os.listdir(source)]

os.system(f"/cr/users/filip/Simulation/QuentinProduction/submit.sh {ADSTs[int(sys.argv[1])]}")

