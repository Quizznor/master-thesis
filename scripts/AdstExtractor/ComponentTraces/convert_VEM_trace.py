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
                station[i][j] = math.floor(bin) / 61.75
    else:
        pass

    return np.array(station)

class RawTrace():

    def __init__(self, file_name : str) -> None :

        # import all necessary files
        self.file_name = file_name

        # do we need background?
        # self.trigger_off_pmt_1 = np.loadtxt(source_path + file_name + "trigger_all_adst_1.csv")
        # self.trigger_off_pmt_2 = np.loadtxt(source_path + file_name + "trigger_all_adst_2.csv")
        # self.trigger_off_pmt_3 = np.loadtxt(source_path + file_name + "trigger_all_adst_3.csv")
        
        self.trigger_on_pmt_1 = np.loadtxt(source_path + file_name + "adst_1.csv")
        self.trigger_on_pmt_2 = np.loadtxt(source_path + file_name + "adst_2.csv")
        self.trigger_on_pmt_3 = np.loadtxt(source_path + file_name + "adst_3.csv")

        # prepare data
        # self.trigger_off_stations = self.trigger_off_pmt_1[0,:]
        # self.trigger_off_pmt_1 = self.trigger_off_pmt_1.T[:,1:]
        # self.trigger_off_pmt_2 = self.trigger_off_pmt_2.T[:,1:]
        # self.trigger_off_pmt_3 = self.trigger_off_pmt_3.T[:,1:]

        # self.trigger_on_stations = self.trigger_on_pmt_1[0,:]
        self.trigger_on_pmt_1 = digitize(self.trigger_on_pmt_1.T, convert = convert_trace)
        self.trigger_on_pmt_2 = digitize(self.trigger_on_pmt_2.T, convert = convert_trace)
        self.trigger_on_pmt_3 = digitize(self.trigger_on_pmt_3.T, convert = convert_trace)

    # split the data into signal and background
    def split_data(self) -> None :

        # bkg = [True if station not in self.trigger_on_stations else False for station in self.trigger_off_stations]
        # sig = [True if station in self.trigger_on_stations else False for station in self.trigger_off_stations]

        # write signal to disk
        for i in range(len(self.trigger_on_pmt_1)):

            with open(target_path + self.file_name[:-1] + ".csv", "a") as signal:
                np.savetxt(signal, self.trigger_on_pmt_1[i], newline=" ", delimiter=" ")
                signal.write("\n")
                np.savetxt(signal, self.trigger_on_pmt_2[i], newline=" ", delimiter=" ")
                signal.write("\n")
                np.savetxt(signal, self.trigger_on_pmt_3[i], newline=" ", delimiter=" ")
                signal.write("\n")

        # # write background to disk
        # for i in range(len(self.trigger_off_pmt_1[bkg])):
        #     with open(target_path + "background/" + self.file_name + f"station-{str(i).zfill(2)}.csv", "w") as background:
        #         np.savetxt(background, self.trigger_off_pmt_1[bkg][i], newline=" ", delimiter=" ")
        #         background.write("\n")
        #         np.savetxt(background, self.trigger_off_pmt_2[bkg][i], newline=" ", delimiter=" ")
        #         background.write("\n")
        #         np.savetxt(background, self.trigger_off_pmt_3[bkg][i], newline=" ", delimiter=" ")
    
        # remove temporary files
        os.remove(source_path + self.file_name + "adst_1.csv")
        os.remove(source_path + self.file_name + "adst_2.csv")
        os.remove(source_path + self.file_name + "adst_3.csv")
        # os.remove(source_path + self.file_name + "trigger_all_adst_1.csv")
        # os.remove(source_path + self.file_name + "trigger_all_adst_2.csv")
        # os.remove(source_path + self.file_name + "trigger_all_adst_3.csv")

if __name__ == "__main__":

    raw_files = np.unique([item[:13] for item in os.listdir(source_path)])

    for file in raw_files:

        try:
            Event = RawTrace(file)
            Event.split_data()

        except (FileNotFoundError, OSError, UserWarning) as exception:
            print(exception)