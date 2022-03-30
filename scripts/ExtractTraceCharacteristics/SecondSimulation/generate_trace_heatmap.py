#!/usr/bin/python3

import sys, os
import numpy as np

for dataset in ["background"]:

    working_dir = f"/cr/data01/filip/second_simulation/tensorflow/{dataset}/"
    signal_files = [working_dir + file for file in os.listdir(working_dir)]
    
    heatmap = np.zeros((600,2048))
    n_tot, max = np.loadtxt(f"/cr/data01/filip/second_simulation/heatmap/{dataset}_binaries.txt")

    for i, file in enumerate(signal_files, 1):

        try:

            i % int(len(signal_files)/100) == 0 and print(f"{i}/{len(signal_files)} done: {(float(i)/len(signal_files) * 100)}%")

            traces = np.loadtxt(file)

            for trace in traces:

                max = np.max(trace) if np.max(trace) > max else max
                indices = [round(bin) if bin > 0 else 0 for bin in trace]

                for i in range(len(trace)):
                    heatmap[indices[i]][i] += 1

                n_tot += 1

        except ValueError:
            print(file)

    with open(f"/cr/data01/filip/second_simulation/heatmap/{dataset}_binaries.txt","w") as file:
        file.write(str(n_tot) + "\n")
        file.write(str(max))

    np.savetxt(f"/cr/data01/filip/second_simulation/heatmap/{dataset}.csv", heatmap)