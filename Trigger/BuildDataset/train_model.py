from Binaries import *

FullScaleDownsampled = EventGenerator("all", apply_downsampling = True, split = 1)

Hardware = HardwareClassifier()
Hardware.make_signal_dataset(FullScaleDownsampled, "fullscale_random_traces_downsampled_scaled")