from Binaries import *

Events = EventGenerator("all", window_length = 360, split = 1)
Trigger = HardwareClassifier()

Trigger.make_signal_dataset(Events, "non_downsample_tot_compatibility")