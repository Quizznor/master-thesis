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

def write_to_csv(storage, data):

    with open(storage,"w") as to_disk:

        for row in data:
            for column in row:
                to_disk.write(str(column) + " ")
            to_disk.write("\n")


def update_array(array, traces):

    for trace in traces:
        # this will fail for too large negative samples (does that ever happen?)
        for i, sample in enumerate(trace):
            j = int(np.round(sample, 1) * 10)

            array[i][j] += 1

    return array

# reduce dataset to simulations where we have trigger on + off
datasets = np.array(os.listdir("/cr/users/filip/data/traces"), dtype = str)
datasets = np.unique([item[:12] for item in datasets])

station_file_location = "/cr/users/filip/condor_output/tmp/station_info.txt"
background_file_location = "/cr/users/filip/condor_output/tmp/cumulative_background.csv"
signal_file_location = "/cr/users/filip/condor_output/tmp/cumulative_signal.csv"   

for step, simulation in enumerate(datasets):

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

    except OSError:
        sys.exit()

    # write data to cumulative dataset(s)
    n_sig, n_bkg = np.loadtxt("/cr/users/filip/condor_output/tmp/station_info.txt", dtype = int, unpack = True)
    n_sig += len(signal_traces)
    n_bkg += len(background_traces)

    with open(station_file_location, "w") as station_info:
        station_info.write(str(n_sig) + "\n" + str(n_bkg))

    background_data = np.loadtxt(background_file_location, dtype = int)
    background_data = update_array(background_data,background_traces)
    write_to_csv(background_file_location, background_data)
        
    signal_data = np.loadtxt(signal_file_location, dtype = int)
    signal_data = update_array(signal_data,signal_traces)
    write_to_csv(signal_file_location, signal_data)

    print(f"finished step {step}/{len(datasets)}")
