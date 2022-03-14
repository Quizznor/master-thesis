#!/usr/bin/python3

import sys, os
import numpy as np
import matplotlib.pyplot as plt

working_dir = "/cr/users/filip/data/second_simulation/tensorflow/signal/"
signal_files = [working_dir + file for file in os.listdir(working_dir)]
# heatmap = np.loadtxt("/cr/users/filip/data/second_simulation/heatmap/signal.csv")
heatmap = np.zeros((600,2048))
n_tot, max = np.loadtxt("/cr/users/filip/data/second_simulation/heatmap/signal_binaries.txt")

for i, file in enumerate(signal_files[:10], 1):

    i % int(len(signal_files)/100) == 0 and print(f"{i}/{len(signal_files)} done: {float(i)/len(signal_files) * 100}%")

    traces = np.loadtxt(file)

    for trace in traces:

        max = np.max(trace) if np.max(trace) > max else max
        indices = [round(bin) if bin > 0 else 0 for bin in trace]

        for i in range(len(trace)):
            heatmap[indices[i]][i] += 1

        n_tot += 1

with open("/cr/users/filip/data/second_simulation/heatmap/signal_binaries.txt","w") as file:
    file.write(str(n_tot) + "\n")
    file.write(str(max))

np.savetxt("/cr/users/filip/data/second_simulation/heatmap/signal.csv", heatmap)