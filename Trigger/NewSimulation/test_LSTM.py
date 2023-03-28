from Binaries import *

# Events = EventGenerator(["19_19.5"], real_background = True)
# training_files = LSTM.get_files("training")
# validation_files = LSTM.get_files("validation")
# Events[0].files, Events[-1].files = training_files, validation_files

LSTM = NNClassifier("120_LSTM_DistinctLayers_FullBandwidth_1Muon")
Events = EventGenerator(LSTM, split = 1)
LSTM.make_signal_dataset(Events, "validation_data_no_cuts")
LSTM.production_test(n_showers = 60000)