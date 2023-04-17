#!/usr/bin/python3

from Binaries import *
ensemble_no = int(sys.argv[1]) + 1


# MockNN = NNClassifier("ENSEMBLES/120_TwoLayer_FullBandwidth_AllEnergies_NoCuts/ensemble_01", supress_print = True)
# Events = EventGenerator([], ignore_low_vem = 0.5)
# Events[0].files += MockNN.get_files("training")
# Events[1].files += MockNN.get_files("validation")

# ThisNN = NNClassifier(f"ENSEMBLES/120_TwoLayer_FullBandwidth_AllEnergies_0_5VEM/ensemble_{ensemble_no}", "two_layer_conv2d")
# ThisNN.train(Events, 10)


ThisNN = NNClassifier(f"ENSEMBLES/120_TwoLayer_FullBandwidth_AllEnergies_0_5VEM/ensemble_{ensemble_no}")
EventsNoCuts = EventGenerator(ThisNN, split = 1)
ThisNN.make_signal_dataset(EventsNoCuts, "validation_data_no_cuts")