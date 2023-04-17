#!/usr/bin/python3

from Binaries import *
ensemble_no = int(sys.argv[1]) + 1

try:
    ThisNN = NNClassifier(f"ENSEMBLES/120_TwoLayer_FullBandwidth_HighEnergies_0_5VEM/ensemble_{ensemble_no:02}")
except FileNotFoundError:
    MockNN = NNClassifier("ENSEMBLES/120_TwoLayer_FullBandwidth_HighEnergies_NoCuts/ensemble_01", supress_print = True)
    Events = EventGenerator([], ignore_low_vem = 0.5)          # cuts will go here
    Events[0].files += MockNN.get_files("training")
    Events[1].files += MockNN.get_files("validation")

    ThisNN = NNClassifier(f"ENSEMBLES/120_TwoLayer_FullBandwidth_HighEnergies_0_5VEM/ensemble_{ensemble_no:02}", "two_layer_conv2d")
    ThisNN.train(Events, 10)

try:
    _ = ThisNN.load_and_print_performance("all_energies_no_cuts")
except FileNotFoundError:
    AllEvents = EventGenerator(":19_19.5", split = 1)
    AllEvents.files += ThisNN.get_files("validation")
    ThisNN.make_signal_dataset(AllEvents, "all_energies_no_cuts")