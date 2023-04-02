from Binaries import *

Trigger = HardwareClassifier("th1")
Events = EventGenerator("all", split = 1, apply_downsampling = True)
Trigger.make_signal_dataset(Events, "th1_per_trace_prediction")