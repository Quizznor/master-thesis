from Binaries import *

LSTM = NNClassifier("120_LSTM_HighEnergy_NoCuts")
# Events = EventGenerator(["19_19.5"], real_background = True)
# training_files = LSTM.get_files("training")
# validation_files = LSTM.get_files("validation")
# Events[0].files, Events[-1].files = training_files, validation_files

_, Events = EventGenerator(["16_16.5", "16.5_17", "17_17.5", "17.5_18", "18_18.5", "18.5_19"])
LSTM.make_signal_dataset(Events, "all_energies")

# TODO: Add energies 19_19.5 to files