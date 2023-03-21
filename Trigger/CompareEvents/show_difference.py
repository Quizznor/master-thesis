from Binaries import *

Trigger = HardwareClassifier()
EventsVEMdownsampled = EventGenerator(["19_19.5"], is_vem = True, apply_downsampling = True, seed = 42, split = 1)
EventsVEMdownsampled.files = ["/cr/tempdata01/filip/QGSJET-II/COMPARE/VEM/" + file for file in os.listdir("/cr/tempdata01/filip/QGSJET-II/COMPARE/VEM")]
Trigger.make_signal_dataset(EventsVEMdownsampled, "VEM_TRACES")
