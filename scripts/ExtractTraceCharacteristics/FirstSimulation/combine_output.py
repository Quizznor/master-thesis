#!/usr/bin/python3

import sys, os, time
import numpy as np

working_directory = "/cr/users/filip/condor_output/"

# while(True):

accumulated_background = np.loadtxt(working_directory + "accumulated_background.csv")
accumulated_signal = np.loadtxt(working_directory + "accumulated_signal.csv")
background_data = os.listdir(working_directory + "/tmp/background/")
signal_data = os.listdir(working_directory + "/tmp/signal/")

# if not background_data and not signal_data:
#     continue

for event in signal_data:
    filename = working_directory + "tmp/signal/" + event
    heatmap = np.loadtxt(filename)
    accumulated_signal += heatmap
    os.remove(filename)

for event in background_data:
    filename = working_directory + "tmp/background/" + event
    heatmap = np.loadtxt(filename)
    accumulated_background += heatmap
    os.remove(filename)

np.savetxt(working_directory + "accumulated_background.csv", accumulated_background)
np.savetxt(working_directory + "accumulated_signal.csv", accumulated_signal)