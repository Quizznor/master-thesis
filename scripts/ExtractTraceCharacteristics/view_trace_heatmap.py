#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})


background = np.loadtxt("/cr/users/filip/data/accumulated_background.csv")
signal = np.loadtxt("/cr/users/filip/data/accumulated_signal.csv")

background = np.log(background)
background[background == -np.inf] = 0

signal = np.log(signal)
signal[signal == -np.inf] = 0

# heatmap for background
plt.figure()
plt.title("Background trace heatmap")
background_heatmap = plt.imshow(background, aspect = 'auto', origin = 'lower')
plt.colorbar(background_heatmap, label="absolute occurence")
plt.ylabel("Signal strength (au)")
plt.xlabel("Time bin (8.3 ns)")
 
# heatmap for signal
plt.figure()
plt.title("Signal trace heatmap")
signal_heatmap = plt.imshow(signal, aspect = 'auto', origin = 'lower')
plt.colorbar(signal_heatmap, label="absulute occurence")
plt.ylabel("Signal strength (au")
plt.xlabel("Time bin (8.3 ns)")

plt.show()