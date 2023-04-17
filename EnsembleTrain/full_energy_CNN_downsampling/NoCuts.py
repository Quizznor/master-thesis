#!/usr/bin/python3

from Binaries import *
ensemble_no = int(sys.argv[1]) + 1


MockNN = NNClassifier("ENSEMBLES/120_TwoLayer_FullBandwidth_AllEnergies_NoCuts/ensemble_01", supress_print = True)
Events = EventGenerator([], apply_downsampling = True, q_peak = np.array([GLOBAL.q_peak_compatibility for _ in range(3)]))
Events[0].files += MockNN.get_files("training")
Events[1].files += MockNN.get_files("validation")

ThisNN = NNClassifier(f"ENSEMBLES/120_TwoLayer_Downsampled_AllEnergies_NoCuts/ensemble_{ensemble_no}", "two_layer_conv2d")
ThisNN.train(Events, 10, apply_downsampling = True)

# EventsNoCuts = EventGenerator(ThisNN, split = 1, apply_downsampling = True, q_peak = np.array([GLOBAL.q_peak_compatibility for _ in range(3)]))
# ThisNN.make_signal_dataset(EventsNoCuts, "validation_data_no_cuts")