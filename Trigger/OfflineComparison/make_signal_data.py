from Binaries import *

Trigger = HardwareClassifier()
Events = EventGenerator("all", split = 1, apply_downsampling = True)
Trigger.make_signal_dataset(Events, "all_triggers_per_trace_prediction")