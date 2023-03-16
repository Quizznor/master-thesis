from Binaries import *

LSTM = NNClassifier("120_LSTM_HighEnergy_NoCuts", "simple_LSTM")
Events = EventGenerator(["19_19.5"], real_background = True)

LSTM.train(Events, 1)