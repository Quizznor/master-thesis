from Binaries import *

vem_peak_scaled = np.array([180.23, 182.52, 169.56]) / 1.115                         # 11.5% surplus

# AllEventsDownsampled = EventGenerator("all", split = 0, real_background = True, apply_downsampling = True, prior = 1,)

# AllEvents = EventGenerator("all", real_background = True,  q_peak = vem_peak_scaled, keep_scale = True, station = "nuria") # prior = 1e-5)  # ignore_low_vem = 1.0)
AllEventsDownsampled = EventGenerator("all", real_background = True,  q_peak = vem_peak_scaled, keep_scale = True, apply_downsampling = True, station = "nuria")
# AllEventsParticleCut = EventGenerator("all", real_background = True, y_peak = vem_peak_scaled, keep_scale = True, station = "nuria", ignore_particles = 1)
# AllEventsNoCut = EventGenerator("all", real_background = True) #, prior = 1e-5) #
# AllEventsParticleCut[-1].unit_test()

# # _ = input("\nPress ENTER to continue")

TestNetwork = NNClassifier("downsampled_vem_peak_scaled", "one_layer_conv2d")
TestNetwork.train(AllEventsDownsampled, 2)

# TestEnsemble = Ensemble("minimal_conv2d_real_background_injections", "one_layer_conv2d")
# TestEnsemble.train(AllEvents, 5)

# make_dataset(TestEnsemble, AllEventsNoCut[-1], "validation_data_no_injections")

# Hardware = HardwareClassifier()
# make_dataset(Hardware, AllEventsDownsampled[-1], "random_traces_downsampled_vem_peak_scaled")

# Hardware.ROC("validation_data")
# TestEnsemble.ROC("validation_data")
# TestEnsemble.ROC("validation_data_no_injections")