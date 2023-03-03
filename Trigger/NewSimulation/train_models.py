from Binaries import *

Assifier = NNClassifier("120_OneLayerNormed_HighEnergyLowPrior_NoCuts", "normed_one_layer_conv2d")
Events = EventGenerator(["19_19.5"], prior = 0.1)

# # Events[-1].physics_test(n_showers = 1000)
Events[-1].training_test(n_showers = 1000)

# Assifier.train(Events, 5)

validation_files = Assifier.get_files("validation")
Events = EventGenerator(["16_16.5", "16.5_17", "17_17.5", "17.5_18", "18_18.5", "18.5_19"], split = 1)
Events.files += validation_files

Assifier.make_signal_dataset(Events, "all_energies")

