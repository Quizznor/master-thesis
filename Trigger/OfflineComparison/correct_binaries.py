from Binaries import *

q_peak = 0.7 * np.array([GLOBAL.q_peak for _ in range(3)])

Events = EventGenerator("/cr/tempdata01/filip/QGSJET-II/LTP/ADST_extracted/", split = 1, apply_downsampling = True)
Trigger = HardwareClassifier()

Trigger.make_signal_dataset(Events, "per_trace_offline_crosscheck")