#!/usr/bin/python3

import sys, os
import numpy as np
import matplotlib.pyplot as plt

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

    # plot traces for given stations
    def plot_station_trace(self, ids):

        traces = self.get_trace_from_stations(ids)

        for id, trace in zip(ids, traces):
            plt.plot(self.time, trace, label = id)  

# reduce dataset to simulations where we have trigger on + off
datasets = np.array(os.listdir("/cr/users/filip/data/traces"), dtype = str)
datasets = np.unique([item[:12] for item in datasets])

total_signal_stations, total_background_stations = 0, 0
n_events = 100

# set up subplot for traces
plt.rcParams.update({'font.size': 22})
fig, (ax1, ax2) = plt.subplots(2, sharex = True)
ax1.set_title(f"Signal traces (appear in *_adst and *_trigger_all_adst), {n_events} events")
ax2.set_title(f"Background traces (appear only in *_trigger_all_adst), {n_events} events")
ax1.set_ylabel("VEM Charge"), ax2.set_ylabel("VEM Charge")
ax2.set_xlabel("Time bins (?? ns)")

for simulation in datasets[:n_events]:

    trigger_on_file = "/cr/users/filip/data/traces/" + simulation + "_adst.csv"
    trigger_off_file = "/cr/users/filip/data/traces/" + simulation + "_trigger_all_adst.csv"

    try:

        trigger_on_data = np.loadtxt(trigger_on_file)
        trigger_off_data = np.loadtxt(trigger_off_file)

    except OSError:
        pass

    TriggerOn = VEMTrace(trigger_on_data)
    TriggerOff = VEMTrace(trigger_off_data)

    # Simply by construction we have signal_stations == TriggerOn.stations
    signal_traces = TriggerOn.traces    # this here is sufficient for signal

    # signal_traces = TriggerOn.get_trace_from_stations(TriggerOff == TriggerOn)        # order matters!
    background_traces = TriggerOff.get_trace_from_stations(TriggerOff != TriggerOn)     # order matters!

    for trace in signal_traces:
        ax1.plot(range(len(trace)), trace, lw = 0.4, c = "steelblue")
        total_signal_stations += 1

    for trace in background_traces:
        ax2.plot(range(len(trace)), trace, lw = 0.4, c = "orange")
        total_background_stations += 1

ax1.plot([],[], c = "steelblue", label = f"{total_signal_stations} stations")
ax2.plot([],[], c = "orange", label = f"{total_background_stations} stations")

ax1.legend()
ax2.legend()
plt.show()    