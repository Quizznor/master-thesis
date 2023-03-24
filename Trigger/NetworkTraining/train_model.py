#!/usr/bin/python3

from Binaries import *
plt.rcParams["text.usetex"] = False

Events = EventGenerator("all", real_background = True, ignore_particles = 1, particle_type = "mu")
Assifier = NNClassifier("120_LSTM_DistinctLayers_FullBandwidth_1Muon_DifferentL", "simple_LSTM")

Assifier.train(Events, 10)
