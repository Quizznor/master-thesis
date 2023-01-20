from Binaries import *

vem_peak_scaled = np.array([180.23, 182.52, 169.56]) / 1.115                         # 11.5% surplus

ParticleCut = EventGenerator("all", real_background = True, station = "Nuria", vem_peak = vem_peak_scaled, keep_scale = True, ignore_particles = 1)
NoCutValidationData = ParticleCut[-1].copy(real_background = True, station = "Nuria", vem_peak = vem_peak_scaled, keep_scale = True)

ParticleCutNetwork = NNClassifier("ignore_1_vem_peak_scaled", "one_layer_conv2d")
ParticleCutNetwork.train(ParticleCut, 5)
make_dataset(ParticleCutNetwork, NoCutValidationData, "no_cut_validation_data")