import matplotlib.pyplot as plt
import numpy as np
import os, sys

fig, axes = plt.subplots(3, sharex = True)
color = {"Component":"b", "Total":"r"}

for type in ["Component","Total"]:
    data_files = [file for file in os.listdir(f"{type}Traces/") if file.endswith(".csv")]

    for i, file in enumerate(data_files):
        vemtrace = np.loadtxt(f"{type}Traces/{file}")
        vemtrace = np.sum(vemtrace, axis = 1)
        axes[i - 3].plot(range(len(vemtrace) - 1), vemtrace[1:], color = color[type], label=type)

    axes[0].set_xlabel("Time bin (8.3 ns)")
    fig.text(0.07, 0.5, 'PMT signal strength (VEM?)', va='center', rotation='vertical')

plt.show()