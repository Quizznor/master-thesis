#!/usr/bin/python3

from Binaries import *
# plt.rcParams["text.usetex"] = False                 # causes Problems on HTCondor

Events = EventGenerator("all", ignore_particles = 1, particle_type = "mu", apply_downsampling = True)
Assifier = NNClassifier("120_LSTM_Downsampled_1Muon", "simple_LSTM")

Assifier.train(Events, 10)
