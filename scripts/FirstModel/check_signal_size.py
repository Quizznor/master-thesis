import sys, os
import numpy as np

working_directory = "/cr/users/filip/data/first_simulation/tensorflow/signal/"
signal_files = os.listdir(working_directory)
n_stations = 0

for file in signal_files:
    data = np.loadtxt(working_directory + file)
    n_stations += data.shape[0]
    print(f"{data.shape} -> n_total = {n_stations}")
