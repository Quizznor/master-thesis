from Binaries import *

for trigger in ["th", "tot", "totd"]:

    Trigger = HardwareClassifier(trigger)
    Events = EventGenerator("all", split = 1)
    Trigger.make_signal_dataset(Events, trigger)