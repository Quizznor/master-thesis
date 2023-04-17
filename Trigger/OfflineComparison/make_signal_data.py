from Binaries import *

q_peak_compatibility = np.array([163.235 for _ in range(3)]) * 0.87
Events = EventGenerator("all", split = 1, apply_downsampling = True, q_peak = q_peak_compatibility)

for trigger in ["th1", "th2", "tot", "totd", "mops"]:
    Trigger = HardwareClassifier(trigger)
    Trigger.make_signal_dataset(Events, f"{trigger}_compatibility_true_vem_conversion")
    Events.__reset__()