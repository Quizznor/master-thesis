#!/usr/bin/python3

from Binaries import *
ensemble_no = int(sys.argv[1]) + 1

event_kwargs = {
                "apply_downsampling" : True, 
                "q_peak" : np.array([GLOBAL.q_peak_compatibility for _ in range(3)]), 
                "sliding_window_length" : 60
                }

try:
    ThisNN = NNClassifier(f"ENSEMBLES/60_LargeCNN_Downsampled_AllEnergies_4_0VEM/ensemble_{ensemble_no:02}")
except FileNotFoundError:
    MockNN = NNClassifier("ENSEMBLES/120_TwoLayer_Downsampled_AllEnergies_NoCuts/ensemble_01", supress_print = True)
    Events = EventGenerator([], **event_kwargs, ignore_low_vem = 4.0)      # cut goes here
    Events[0].files += MockNN.get_files("training")
    Events[1].files += MockNN.get_files("validation")

    ThisNN = NNClassifier(f"ENSEMBLES/60_eCNN_Downsampled_AllEnergies_4_0VEM/ensemble_{ensemble_no:02}", "60_CNN")
    ThisNN.train(Events, 10, apply_downsampling = True)

try:
    _ = ThisNN.load_and_print_performance("validation_data_no_cuts")
except FileNotFoundError:
    EventsNoCut = EventGenerator(ThisNN, split = 1, **event_kwargs)
    ThisNN.make_signal_dataset(EventsNoCut, "validation_data_no_cuts")