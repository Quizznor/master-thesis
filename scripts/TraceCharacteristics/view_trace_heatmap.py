#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})


background = np.loadtxt("/cr/users/filip/condor_output/tmp/cumulative_background.csv", dtype = int)
signal = np.loadtxt("/cr/users/filip/condor_output/tmp/cumulative_signal.csv", dtype = int)
n_sig, n_bkg = np.loadtxt("/cr/users/filip/condor_output/tmp/station_info.txt", dtype = int, unpack = True)

# heatmap for background
plt.figure()
background_heatmap = plt.imshow(background.T, aspect = 'auto', cmap = 'viridis', origin = 'lower')

plt.title(f"Background trace heatmap, {n_bkg} stations")
plt.colorbar(background_heatmap, label="relative occurence (au)")
plt.ylabel("VEM charge (0.1 ADC counts)")
plt.xlabel("Time bin (8.3 ns)")
 
# heatmap for signal
plt.figure()
signal_heatmap = plt.imshow(signal.T, aspect = 'auto', cmap = 'viridis', origin = 'lower')

plt.title(f"Background trace heatmap, {n_sig} stations")
plt.colorbar(signal_heatmap, label="relative occurence (au)")
plt.ylabel("VEM charge (0.1 ADC counts)")
plt.xlabel("Time bin (8.3 ns)")

plt.show()