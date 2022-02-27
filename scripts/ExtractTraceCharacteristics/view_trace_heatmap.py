#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})


background = np.loadtxt("/cr/users/filip/data/first_simulation/heatmap/accumulated_background.csv")
signal = np.loadtxt("/cr/users/filip/data/first_simulation/heatmap/accumulated_signal.csv")

background = np.log(background)
background[background == -np.inf] = 0

signal = np.log(signal)
signal[signal == -np.inf] = 0

# heatmap for background
plt.figure()
plt.title("Background trace heatmap, 20281 events")
background_heatmap = plt.imshow(background, aspect = 'auto', origin = 'lower', extent = [0, 2048, 0, 22])
plt.colorbar(background_heatmap, label="log(absolute occurence)")
plt.ylabel("Signal strength (VEM)")
plt.xlabel("Time bin (8.3 ns)")
 
# heatmap for signal
plt.figure()
plt.title("Signal trace heatmap, 20281 events")
signal_heatmap = plt.imshow(signal, aspect = 'auto', origin = 'lower', extent = [0, 2048, 0, 600])
plt.colorbar(signal_heatmap, label="log(absolute occurence)")
plt.ylabel("Signal strength (VEM)")
plt.xlabel("Time bin (8.3 ns)")

plt.show()