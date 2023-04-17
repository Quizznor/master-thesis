#!/usr/bin/python3

from Binaries import *
plt.rcParams["text.usetex"] = False

# set up training environment
Events = EventGenerator("19_19.5", ignore_particles = 8, particle_type = "mu")
Assifier = Ensemble("120_TwoLayer_FullBandwidth_HighEnergies_8Muon", "two_layer_conv2d", n_models = 7)

# train the classifier
Assifier.train(Events, 10)

# # create predictions for the whole dataset
# AllEvents = EventGenerator(":19_19.5", split = 1)
# ValFiles = EventGenerator(Assifier, real_background = True, split = 1)
# AllEvents.files += ValFiles.files
# Assifier.make_signal_dataset(AllEvents, "all_energies_no_cuts")