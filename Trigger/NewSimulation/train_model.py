from Binaries import *

vem_peak_scaled = np.array([180.23, 182.52, 169.56]) / 1.115                         # 11.5% surplus

ParticleCut = EventGenerator("all", real_background = True, station = "Nuria", 
                            vem_peak = vem_peak_scaled, keep_scale = True, 
                            window_length = 360)
# NoCutValidationData = ParticleCut[-1].copy(real_background = True, station = "Nuria", vem_peak = vem_peak_scaled, keep_scale = True)

NetworkOneLayer = NNClassifier("360_input_one_layer_vem_peak_scaled", "one_layer_downsampling_equal")
NetworkTwoLayer = NNClassifier("360_input_two_layer_vem_peak_scaled", "two_layer_downsampling_equal")
NetworkOneLayer.train(ParticleCut, 5)
NetworkTwoLayer.train(ParticleCut, 5)

# make_dataset(ParticleCutNetwork, NoCutValidationData, "no_cut_validation_data")