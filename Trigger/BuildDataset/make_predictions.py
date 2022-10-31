from Binaries import *

TriggerClassifier = HardwareClassifier()
# Ensemble1 = Ensemble("minimal_conv2d_real_background")
# _, RandomTraces = EventGenerator("all", real_background = True, apply_downsampling = False)
_, RandomTracesDownsampled = EventGenerator("all", real_background = True, apply_downsampling = True)

RandomTracesDownsampled.unit_test(1000)

# make_dataset(Ensemble1, RandomTraces, "validation_data")
# make_dataset(TriggerClassifier, RandomTracesDownsampled, "random_traces_downsampled")