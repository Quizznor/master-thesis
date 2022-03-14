#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})


signal = np.loadtxt("/cr/users/filip/data/second_simulation/heatmap/signal.csv")
n_tot, max = np.loadtxt("/cr/users/filip/data/second_simulation/heatmap/signal_binaries.txt")

signal = np.log(signal)
signal[signal == -np.inf] = 0
 
# heatmap for signal
plt.figure()
plt.title(f"Signal trace heatmap, {int(n_tot)} traces")
signal_heatmap = plt.imshow(signal, aspect = 'auto', origin = 'lower', extent = [0, 2048, 0, 600])
plt.colorbar(signal_heatmap, label="log(absolute occurence)")
plt.ylabel("Signal strength (VEM)")
plt.xlabel("Time bin (8.3 ns)")

plt.show()