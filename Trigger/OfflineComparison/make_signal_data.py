from Binaries import *

q_peak_compatibility = np.array([163.235 for _ in range(3)])
Events = EventGenerator("all", split = 1, apply_downsampling = True, q_peak = q_peak_compatibility)

for trigger in ["th2", "tot", "totd"]:
    Trigger = HardwareClassifier(trigger)
    Trigger.make_signal_dataset(Events, f"{trigger}_q_peak_compatibility")
    Events.__reset__()