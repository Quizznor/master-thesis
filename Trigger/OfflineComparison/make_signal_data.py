from Binaries import *

for trigger in ["th", "tot", "totd"]:

    Trigger = HardwareClassifier(trigger)
    Events = EventGenerator("all", split = 1, apply_downsampling = True)
    Trigger.make_signal_dataset(Events, trigger + "_per_trace_prediction")