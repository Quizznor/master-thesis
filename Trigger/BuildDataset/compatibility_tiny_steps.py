from Binaries import *

q_peak_compatibility = np.array([163.235 for _ in range(3)])
Events = EventGenerator("all", split = 1, real_background = False, apply_downsampling = True, sigma = 0, window_step = 1, q_peak = q_peak_compatibility)

Trigger = HardwareClassifier()
Trigger.make_signal_dataset(Events, "compatibility_tiny_steps")