#!/usr/bin/python3

import numpy as np
import warnings
import sys, os
import math

warnings.filterwarnings("error")

source_path = sys.argv[1]
target_path = sys.argv[2]
convert_trace = True if sys.argv[3] == "True" else False

def digitize(station, convert):

    if convert:
        for i, trace in enumerate(station):
            for j, bin in enumerate(trace):
                station[i][j] = math.floor(bin) / 215
    else:
        pass

    return np.array(station)

class RawTrace():

    def __init__(self, file_name : str) -> None :

        # import all necessary files
        self.file_name = file_name
        
        self.trigger_on_pmt_1 = np.loadtxt(source_path + file_name + "_1.csv")
        self.trigger_on_pmt_2 = np.loadtxt(source_path + file_name + "_2.csv")
        self.trigger_on_pmt_3 = np.loadtxt(source_path + file_name + "_3.csv")

        print(self.trigger_on_pmt_1.shape, self.trigger_on_pmt_2.shape, self.trigger_on_pmt_3.shape)

        # self.trigger_on_stations = self.trigger_on_pmt_1[0,:]
        self.trigger_on_pmt_1 = digitize(self.trigger_on_pmt_1.T, convert = convert_trace)
        self.trigger_on_pmt_2 = digitize(self.trigger_on_pmt_2.T, convert = convert_trace)
        self.trigger_on_pmt_3 = digitize(self.trigger_on_pmt_3.T, convert = convert_trace)

        print(self.trigger_on_pmt_1.shape, self.trigger_on_pmt_2.shape, self.trigger_on_pmt_3.shape)

    # split the data into signal and background
    def split_data(self) -> None :

        # write signal to disk
        for i in range(len(self.trigger_on_pmt_1)):

            with open(target_path + self.file_name + ".csv", "a") as signal:
                np.savetxt(signal, self.trigger_on_pmt_1[i], newline=" ", delimiter=" ")
                signal.write("\n")
                np.savetxt(signal, self.trigger_on_pmt_2[i], newline=" ", delimiter=" ")
                signal.write("\n")
                np.savetxt(signal, self.trigger_on_pmt_3[i], newline=" ", delimiter=" ")
                signal.write("\n")

if __name__ == "__main__":

    raw_files = np.unique([item[:9] for item in os.listdir(source_path)])

    for file in raw_files:

        try:
            Event = RawTrace(file)
            Event.split_data()

        except (FileNotFoundError, OSError, UserWarning) as exception:
            print(exception)