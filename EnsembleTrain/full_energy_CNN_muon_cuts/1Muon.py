#!/usr/bin/python3

from Binaries import *
ensemble_no = int(sys.argv[1]) + 1

try:
    ThisNN = NNClassifier(f"ENSEMBLES/120_TwoLayer_FullBandwidth_AllEnergies_1Muon/ensemble_{ensemble_no:02}")
except FileNotFoundError:

    MockNN = NNClassifier("ENSEMBLES/120_TwoLayer_FullBandwidth_AllEnergies_NoCuts/ensemble_01", supress_print = True)
    Events = EventGenerator([], ignore_particles = 1, particle_type = "mu")
    Events[0].files += MockNN.get_files("training")
    Events[1].files += MockNN.get_files("validation")

    ThisNN = NNClassifier(f"ENSEMBLES/120_TwoLayer_FullBandwidth_AllEnergies_1Muon/ensemble_{ensemble_no:02}", "two_layer_conv2d")
    ThisNN.train(Events, 10)

try:
    _ = ThisNN.load_and_print_performance("validation_data_no_cuts")
except FileNotFoundError:
    EventsNoCuts = EventGenerator(ThisNN, split = 1)
    ThisNN.make_signal_dataset(EventsNoCuts, "validation_data_no_cuts")