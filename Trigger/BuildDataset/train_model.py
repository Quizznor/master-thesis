from Binaries import *

# vem_peak_scaled = np.array([180.23, 182.52, 169.56]) / 1.115                         # 11.5% surplus

# AllEventsDownsampled = EventGenerator("all", split = 0, real_background = True, apply_downsampling = True, prior = 1,)

# AllEvents = EventGenerator("all", real_background = True,  q_peak = vem_peak_scaled, keep_scale = True, station = "nuria", window_size = 360) # prior = 1e-5)  # ignore_low_vem = 1.0)
# AllEventsDownsampled = EventGenerator("all", real_background = True, keep_scale = True, q_peak = vem_peak_scaled, apply_downsampling = True, station = "nuria")
# AllEventsParticleCut = EventGenerator("all", real_background = True, y_peak = vem_peak_scaled, keep_scale = True, station = "nuria", ignore_particles = 1)
# AllEventsNoCut = EventGenerator("all", real_background = True) #, prior = 1e-5) #
# AllEventsParticleCut[-1].unit_test()

# # _ = input("\nPress ENTER to continue")

# OneLayerDownsamplingEqual = NNClassifier("downsampling_equal_vem_peak_scaled_1_layer", "one_layer_downsampling_equal")
# TwoLayerDownsamplingEqual = NNClassifier("downsampling_equal_vem_peak_scaled_2_layer", "two_layer_downsampling_equal")

# OneLayerDownsamplingEqual.train(AllEvents, 5)
# TwoLayerDownsamplingEqual.train(AllEvents, 5)

# TestEnsemble = Ensemble("minimal_conv2d_real_background_injections", "one_layer_conv2d")
# TestEnsemble.train(AllEvents, 5)

# make_dataset(TestEnsemble, AllEventsNoCut[-1], "validation_data_no_injections")

FullScaleDownsampled = EventGenerator("all", apply_downsampling = True, split = 1)

Hardware = HardwareClassifier()
make_dataset(Hardware, FullScaleDownsampled, "fullscale_downsampled_old_algorithm")

# Hardware.ROC("validation_data")
# TestEnsemble.ROC("validation_data")
# TestEnsemble.ROC("validation_data_no_injections")