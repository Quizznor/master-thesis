#!/usr/bin/python3

# # first simulation
# from Binaries import *
# plt.rcParams["text.usetex"] = False

# Events = EventGenerator("all")
# CNN = Ensemble("120_TwoLayer_FullBandwidth_AllEnergies_NoCuts", "two_layer_conv2d")
# CNN.train(Events, 10)

# # second simulation
from Binaries import *
plt.rcParams["text.usetex"] = False

EventsHE = EventGenerator("19_19.5")
CNN = Ensemble("120_TwoLayer_FullBandwidth_HighEnergies_NoCuts", "two_layer_conv2d")
CNN.train(EventsHE, 10)

AllEvents = EventGenerator(":19_19.5", split = 1)
ValFiles = EventGenerator(CNN, real_background = False, split = 1)
AllEvents.files += ValFiles.files

CNN.make_signal_dataset(AllEvents, "all_energies")