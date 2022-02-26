#!/usr/bin/python3

import numpy as np

signal_data = np.zeros((2048, 7000), dtype = int)     # 7000 rows = 0.1 ADC resolution with max 700 VEM
background_data = np.zeros((2048, 100), dtype = int)  # 100 rows  = 0.1 ADC resolution with max 10 VEM

with open("/cr/users/filip/condor_output/tmp/cumulative_background.csv","w") as to_disk:

    for row in background_data:
        for column in row:
            to_disk.write(str(column) + " ")
        
        to_disk.write("\n")

with open("/cr/users/filip/condor_output/tmp/cumulative_signal.csv","w") as to_disk:

    for row in signal_data:
        for column in row:
            to_disk.write(str(column) + " ")
        
        to_disk.write("\n")

with open("/cr/users/filip/condor_output/tmp/station_info.txt", "w") as to_disk:
    to_disk.write("0\n0") 