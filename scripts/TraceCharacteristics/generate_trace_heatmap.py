#!/usr/bin/python3

import sys, os
import numpy as np

class VEMTrace():

    def __init__(self, data):

        self.stations = data[0, :]  # station ids
        self.traces = data.T[:,1:]  # vem traces
        self.time = range(2048)     # time bins

    # return station ids common with other object 
    def __eq__(self, other):
        return [station for station in self.stations if station in other.stations]

    # return stations exclusive to this object
    def __ne__(self, other):
        return [station for station in self.stations if station not in other.stations]

    # return traces for given station ids
    def get_trace_from_stations(self, ids):

        traces = []

        for id in ids:
            traces.append(self.traces[np.where(self.stations == id)[0][0]])
        
        return np.array(traces)

# reduce dataset to simulations where we have trigger on + off
datasets = np.array(os.listdir("/cr/users/filip/data/traces"), dtype = str)
datasets = np.unique([item[:12] for item in datasets])

# stepsize = 500
# start = int(sys.argv[1])      * stepsize
# stop = (int(sys.argv[1]) + 1) * stepsize

# try:
#     simulations = datasets[start:stop]
# except IndexError:
#     simulations = datasets[start:-1]

signal_bins = 600
background_bins = 100
time_bins = 2048

heatmap_background = np.zeros((background_bins, time_bins))
heatmap_signal = np.zeros((signal_bins, time_bins))

for i, simulation in enumerate(datasets):

    i % 100 == 0 and print(f"Calculating step {i}/20281")

    trigger_on_file = "/cr/users/filip/data/traces/" + simulation + "_adst.csv"
    trigger_off_file = "/cr/users/filip/data/traces/" + simulation + "_trigger_all_adst.csv"

    try:

        trigger_on_data = np.loadtxt(trigger_on_file)
        trigger_off_data = np.loadtxt(trigger_off_file)

        TriggerOn = VEMTrace(trigger_on_data)
        TriggerOff = VEMTrace(trigger_off_data)

        # Simply by construction we have signal_stations == TriggerOn.stations
        signal_traces = TriggerOn.traces    # this here is sufficient for signal

        # signal_traces = TriggerOn.get_trace_from_stations(TriggerOff == TriggerOn)        # order matters!
        background_traces = TriggerOff.get_trace_from_stations(TriggerOff != TriggerOn)     # order matters!

        for trace in signal_traces:
            bins, _, _ = np.histogram2d(trace, TriggerOn.time, bins = (signal_bins, time_bins))
            heatmap_signal += bins

        for trace in background_traces:
            bins, _, _ = np.histogram2d(trace, TriggerOff.time, bins = (background_bins, time_bins))
            heatmap_background += bins


    except OSError:
        continue

np.savetxt(f"/cr/users/filip/condor_output/accumulated_background.csv", heatmap_background)
np.savetxt(f"/cr/users/filip/condor_output/accumulated_signal.csv", heatmap_signal)
