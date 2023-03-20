#!/usr/bin/python3

from Binaries import *

Assifier = NNClassifier("120_LSTM_FullBandwidth_NoCuts", "simple_LSTM")
Events = EventGenerator("all")
Assifier.train(Events, 10)
