from Binaries import *

# Assifier = NNClassifier("120_OneLayer_HighEnergyDownsampled_NoCuts")

# q_peak_scaled = np.array([0.3 * GLOBAL.q_peak for _ in range(3)])
# q_charge_scaled = np.array([0.3 * GLOBAL.q_charge for _ in range(3)])


q_peak = np.array([1, 1, 1])
Trigger = HardwareClassifier()
Events = EventGenerator(["19_19.5"], apply_downsampling = True, seed = 42, q_peak = q_peak, split = 1)
Trigger.make_signal_dataset(Events, "VEM_TRACES")
Trigger.spd_energy_efficiency("VEM_TRACES")


vem_dir = "/cr/tempdata01/filip/QGSJET-II/COMPARE/VEM/"
Events.files = [vem_dir + file for file in os.listdir(vem_dir)]

Events.physics_test(n_showers = 1000)

# Events[-1].training_test(n_showers = 1000)
# Assifier.train(Events, 10)

# validation_files = Assifier.get_files("validation")
# Events = EventGenerator(":19_19.5", split = 1)
# Events.files += validation_files


