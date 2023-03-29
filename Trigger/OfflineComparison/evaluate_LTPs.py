from Binaries import *

Trigger = HardwareClassifier()
Events = EventGenerator("/cr/tempdata01/filip/QGSJET-II/LTP/ADST_extracted/", real_background = False, split = 1)

Trigger.make_signal_dataset(Events, "offline_cross_check", save_traces = True)