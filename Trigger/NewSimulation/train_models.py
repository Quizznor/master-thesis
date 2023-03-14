from Binaries import *

# Assifier = NNClassifier("120_OneLayer_HighEnergyDownsampled_NoCuts")

Assifier = HardwareClassifier()
Events = EventGenerator("all", apply_downsampling = True, seed = 42)
Events[-1].physics_test(n_showers = 1000)

# Events[-1].training_test(n_showers = 1000)
# Assifier.train(Events, 10)

# validation_files = Assifier.get_files("validation")
# Events = EventGenerator(":19_19.5", split = 1)
# Events.files += validation_files

# Assifier.make_signal_dataset(Events, "random_traces_downsampled_scaled")

