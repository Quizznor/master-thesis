from Binaries import *

# _, TestEvents = EventGenerator("all", apply_downsampling = True)
TestClassifier = HardwareClassifier()
# TestClassifier.make_signal_dataset(TestEvents, "random_traces_downsampled_scaled", save_traces = False)

# TestClassifier.spd_energy_efficiency("random_traces_downsampled_scaled")
TestClassifier.do_t3_simulation("random_traces_downsampled_scaled")