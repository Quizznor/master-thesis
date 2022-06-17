#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})


signal = np.loadtxt("/cr/data01/filip/second_simulation/heatmap/signal.csv")
n_tot_sig, max_sig = np.loadtxt("/cr/data01/filip/second_simulation/heatmap/signal_binaries.txt")

background = np.loadtxt("/cr/data01/filip/second_simulation/heatmap/background.csv")
n_tot_bkg, max_bkg = np.loadtxt("/cr/data01/filip/second_simulation/heatmap/background_binaries.txt")

signal = np.log(signal)
signal[signal == -np.inf] = 0

background = np.log(background)
background[background == -np.inf] = 0
 
# heatmap for signal
plt.figure()
plt.title(f"Signal trace heatmap, {int(n_tot_sig)} traces")
signal_heatmap = plt.imshow(signal, aspect = 'auto', origin = 'lower', extent = [0, 2048, 0, 600])
plt.colorbar(signal_heatmap, label="log(absolute occurence)")
plt.ylabel("Signal strength (VEM)")
plt.xlabel("Time bin (8.3 ns)")

# heatmap for background
plt.figure()
plt.title(f"Signal trace heatmap, {int(n_tot_sig)} traces")
background_heatmap = plt.imshow(background, aspect = 'auto', origin = 'lower', extent = [0, 2048, 0, 600])
plt.colorbar(background_heatmap, label="log(absolute occurence)")
plt.ylabel("Signal strength (VEM)")
plt.xlabel("Time bin (8.3 ns)")

plt.show()