#!/usr/bin/python3

import sys, os
import numpy as np

working_directory = "/cr/users/filip/data/first_simulation/"
dataset = [file for file in os.listdir(working_directory + "traces/") if not file.endswith("_trigger_all_adst.csv")]
dataset_fragments = np.array_split(dataset, 100)


with open(working_directory + "signal-" + sys.argv[1].zfill(3) + ".csv", "w") as storage:

    # split data into 100 files with ~50MB diskspace each
    for file in dataset_fragments[int(sys.argv[1])]:

        data = np.loadtxt(working_directory + "traces/" + file, dtype = str).T

        # data[:, 0] is the station ID, not important here!
        for trace in data[:,1:]:
            storage.write(" ".join(trace))
            storage.write("\n")