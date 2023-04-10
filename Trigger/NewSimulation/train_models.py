from Binaries import *

q_peak_compatibility = np.array([GLOBAL.q_peak_compatibility for _ in range(3)]) * (1 - 0.133)
Events = EventGenerator("all", apply_downsampling = True, split = 1, q_peak = q_peak_compatibility)

Trigger = HardwareClassifier()
Trigger.make_signal_dataset(Events, "compatibility_vem_corrected")