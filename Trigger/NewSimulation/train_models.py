from Binaries import *

# Assifier = NNClassifier("120_OneLayer_HighEnergyDownsampled_NoCuts")

q_peak_scaled = np.array([0.3 * GLOBAL.q_peak for _ in range(3)])
q_charge_scaled = np.array([0.3 * GLOBAL.q_charge for _ in range(3)])

Assifier = HardwareClassifier()
Events = EventGenerator("all", apply_downsampling = True, seed = 42, q_peak = q_peak_scaled)
# Events.physics_test(n_showers = 1000)

# Events[-1].training_test(n_showers = 1000)
# Assifier.train(Events, 10)

# validation_files = Assifier.get_files("validation")
# Events = EventGenerator(":19_19.5", split = 1)
# Events.files += validation_files

Assifier.make_signal_dataset(Events[-1], "random_traces_downsampled_vem_0.300")

