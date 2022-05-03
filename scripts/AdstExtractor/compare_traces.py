import matplotlib.pyplot as plt
import numpy as np
import os, sys

plt.rcParams.update({'font.size': 18})
color = {"Component":"b", "VEM":"r"}
linestyle = {"Component":"-", "VEM":":"}
ratios = [[],[]]

for i, type in enumerate(["Component","VEM"]):
    data_files = [file for file in os.listdir(f"{type}Traces/") if file.endswith(".csv")]

    factor = 215 / 61.75 if type == "VEM" else 1

    for i, file in enumerate(data_files):
        vemtraces = np.loadtxt(f"{type}Traces/{file}")
        print(vemtraces.shape)

        for trace in vemtraces[:6]:
            plt.plot(range(len(trace) - 1), factor * trace[1:], color = color[type], ls = linestyle[type])

plt.ylabel('PMT signal strength (VEM_peak)')
plt.xlabel("Time bin (8.3 ns)")

# plt.set_title(f"PMT #{i}")
plt.plot([],[], c="r", label = "recStation.GetVEMTrace(PMT)", ls = ":")
plt.plot([],[], c="b", label = r"converted $\Sigma$ (photon, electron, muon)", ls = "-")
plt.legend()

plt.show()