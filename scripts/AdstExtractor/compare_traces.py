import matplotlib.pyplot as plt
import numpy as np
import os, sys

fig, axes = plt.subplots(3, sharex = True)
plt.title("Station wise comparison of traces")
plt.rcParams.update({'font.size': 20})
color = {"Component":"b", "Total":"r"}
linestyle = {"Component":"-", "Total":":"}

for type in ["Component","Total"]:
    data_files = [file for file in os.listdir(f"{type}Traces/") if file.endswith(".csv")]

    for i, file in enumerate(data_files):
        vemtraces = np.loadtxt(f"{type}Traces/{file}", unpack = True)

        for trace in vemtraces:
            axes[i - 3].plot(range(len(trace) - 1), trace[1:], color = color[type], ls = linestyle[type])

fig.text(0.07, 0.5, 'PMT signal strength', va='center', rotation='vertical')
axes[-1].set_xlabel("Time bin (8.3 ns)")

for i in range(len(axes)):
    axes[i].set_title(f"PMT #{i}")
    axes[i].plot([],[], c="r", label = "Total trace", ls = ":")
    axes[i].plot([],[], c="b", label = r"$\Sigma$ (photon, electron, muon)", ls = "-")
    axes[i].legend()

plt.show()