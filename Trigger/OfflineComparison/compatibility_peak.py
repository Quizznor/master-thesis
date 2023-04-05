from Binaries import *

q_peak_compatibility = np.array([163.235 for _ in range(3)])

Trigger = HardwareClassifier()
Events = EventGenerator("all", split = 1, apply_downsampling = True, q_peak = q_peak_compatibility)
Events.physics_test(1000)
Trigger.make_signal_dataset(Events, "q_peak_compatibility")