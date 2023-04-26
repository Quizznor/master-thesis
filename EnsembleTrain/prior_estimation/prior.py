#!/usr/bin/python3

from Binaries import *
# ensemble_no = int(sys.argv[1]) + 1
priors = np.linspace(0.05, 0.95, 10)
ensemble_no = int(sys.argv[1])

event_kwargs = {
                "apply_downsampling" : True, 
                "q_peak" : np.array([GLOBAL.q_peak_compatibility for _ in range(3)]), 
                "prior" : np.round(priors[ensemble_no], 2)
                }

try:
    ThisNN = NNClassifier(f"120_eCNN_Downsampled_AllEnergies_p{event_kwargs['prior']}")
except FileNotFoundError:
    MockNN = NNClassifier("ENSEMBLES/120_TwoLayer_Downsampled_AllEnergies_NoCuts/ensemble_01", supress_print = True)
    Events = EventGenerator([], **event_kwargs, ignore_low_vem = 2.0)      # cut goes here
    Events[0].files += MockNN.get_files("training")
    Events[1].files += MockNN.get_files("validation")

    ThisNN = NNClassifier(f"120_eCNN_Downsampled_AllEnergies_p{event_kwargs['prior']}", "120_CNN")
    ThisNN.train(Events, 10, apply_downsampling = True)

try:
    _ = ThisNN.load_and_print_performance("validation_data_no_cuts")
except FileNotFoundError:
    EventsNoCut = EventGenerator(ThisNN, split = 1, **event_kwargs)
    ThisNN.make_signal_dataset(EventsNoCut, "validation_data_no_cuts")