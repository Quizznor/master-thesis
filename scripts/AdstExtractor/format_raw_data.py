#!/usr/bin/python3

import numpy as np
import sys, os

callback_list = "/cr/users/filip/data/second_simulation/tensorflow/failed_events.csv"
source_path = "/cr/users/filip/data/second_simulation/tensorflow/tmp_data/"
raw_files = np.unique([item[:13] for item in os.listdir(source_path)])

class RawTrace():

    def __init__(file_name : str) -> None :

        # import all necessary files
        try:
            trigger_off_pmt_1 = np.loadtxt(source_path + file_name + "trigger_all_1.csv")
            trigger_off_pmt_2 = np.loadtxt(source_path + file_name + "trigger_all_2.csv")
            trigger_off_pmt_3 = np.loadtxt(source_path + file_name + "trigger_all_3.csv")
            
            trigger_on_pmt_1 = np.loadtxt(source_path + file_name + "adst_1.csv")
            trigger_on_pmt_2 = np.loadtxt(source_path + file_name + "adst_2.csv")
            trigger_on_pmt_3 = np.loadtxt(source_path + file_name + "adst_3.csv")

        except FileNotFoundError:
            with open(callback_list, "a") as callback:
                callback_list.write(file_name + "\n")

        # prepare data
        trigger_off_stations = trigger_off_pmt_1[0,:]
        trigger_off_pmt_1 = trigger_off_pmt_1.T[:,1:]
        trigger_off_pmt_2 = trigger_off_pmt_2.T[:,1:]
        trigger_off_pmt_3 = trigger_off_pmt_3.T[:,1:]

        trigger_on_stations = trigger_on_pmt_1[0,:]
        trigger_on_pmt_1 = trigger_on_pmt_1.T[:,1:]
        trigger_on_pmt_2 = trigger_on_pmt_2.T[:,1:]
        trigger_on_pmt_3 = trigger_on_pmt_3.T[:,1:]

    # split the data into signal and background
    def split_data(self) -> None :
        # TODO !!
        pass