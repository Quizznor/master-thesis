from Binaries import *

# pt.PRC("large_model_validation_data", c = "yellow", ls = "solid") # -> trash
# pt.PRC("small_model_validation_data", c = "green", ls = "solid") # -> trash

# # model background
# PRC("current_trigger_validation_data", ls = "solid", c = "k")
# PRC("one_layer_conv2d_cut_0.00VEM_validation_data", c = "orange", ls = "solid")
# # PRC("one_layer_conv2d_cut_0.20VEM_validation_data", c = "darkblue", ls = "solid")
# # PRC("one_layer_conv2d_cut_0.50VEM_validation_data", c = "blue", ls = "solid")
# # PRC("one_layer_conv2d_cut_1.00VEM_validation_data", c = "lightblue", ls = "solid")
# # PRC("one_layer_conv2d_cut_2.00VEM_validation_data", c = "yellow", ls = "solid")
# # PRC("one_layer_conv2d_cut_4.63VEM_validation_data", c = "green", ls = "solid")
# PRC("one_layer_conv2d_cut_10.0VEM_validation_data", c = "brown", ls = "solid")
# # PRC("bayes_m249_validation_data", c = "darksalmon", ls = "solid")
# # PRC("bayes_m244_validation_data", c = "lime", ls = "solid")
# # PRC("bayes_LHQ_validation_data", c = "steelblue", ls = "solid")

# print()

# # model background
# PRC("current_trigger_random_traces", ls = "--", c = "k")
# PRC("one_layer_conv2d_cut_0.00VEM_random_traces", c = "orange", ls = "--")
# # PRC("one_layer_conv2d_cut_0.20VEM_random_traces", c = "darkblue", ls = "--")
# # PRC("one_layer_conv2d_cut_0.50VEM_random_traces", c = "blue", ls = "--")
# # PRC("one_layer_conv2d_cut_1.00VEM_random_traces", c = "lightblue", ls = "--")
# # PRC("one_layer_conv2d_cut_2.00VEM_random_traces", c = "yellow", ls = "--")
# # PRC("one_layer_conv2d_cut_4.63VEM_random_traces", c = "green", ls = "--")
# PRC("one_layer_conv2d_cut_10.0VEM_random_traces", c = "brown", ls = "--")
# # PRC("bayes_m249_random_traces", c = "darksalmon", ls = "--")
# # PRC("bayes_m244_random_traces", c = "lime", ls = "--")
# # PRC("bayes_LHQ_random_traces", c = "steelblue", ls = "--")

# plt.legend(ncol = 2)

# fig = plt.figure()
# ax1 = plt.subplot(2,1,1)
# spd_energy("one_layer_conv2d_cut_4.63VEM_validation_data", label = "NN, 4.63 VEM cut")
# ax1.set_xlabel(""), ax1.set_xticks([],[])
# plt.legend()
# plt.subplot(2,1,2)
# spd_energy("current_trigger_validation_data", marker = "s", ls = "--", label = "Hardware")
# plt.legend()

# import matplotlib


# cbar_ax = fig.add_axes([0.87, 0.1, 0.05, 0.8])
# cmap = cmap.plasma
# bounds = [16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5]
# norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

# fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
#              cax=cbar_ax,
#              label="log E")

CurrentTrigger = HardwareClassifier()
# NetworkNoCut = NNClassifier("minimal_conv2d_cut_0.00VEM/model_converged", supress_print = True)

print("\nDATASET".ljust(72) + "TP      FP      TN      FN     sum")
ROC(CurrentTrigger, "random_traces", c = "k", ls = "--")

# print()

# PRC(NetworkNoCut, "validation_data", c = "green", ls = "--")
# PRC(NetworkNoCut, "validation_data_downsampled", c = "green")

# NetworkNoCut = NNClassifier("minimal_conv2d_0.00VEM_downsampled/model_3", supress_print = True)
# PRC(NetworkNoCut, "validation_data", c = "green", ls = ":")

# print()

# NetworkFifthCut = NNClassifier("minimal_conv2d_cut_0.20VEM/model_converged", supress_print = True)
# PRC(NetworkFifthCut, "validation_data", c = "steelblue", ls = "--")
# PRC(NetworkFifthCut, "validation_data_downsampled", c = "steelblue")

# NetworkFifthCut = NNClassifier("minimal_conv2d_0.20VEM_downsampled/model_3", supress_print = True)
# PRC(NetworkFifthCut, "validation_data", c = "steelblue", ls = ":")

# print()

# NetworkHalfCut = NNClassifier("minimal_conv2d_cut_0.50VEM/model_converged", supress_print = True)
# PRC(NetworkHalfCut, "validation_data", c = "yellow", ls = "--")
# PRC(NetworkHalfCut, "validation_data_downsampled", c = "yellow")

# NetworkHalfCut = NNClassifier("minimal_conv2d_0.50VEM_downsampled/model_3", supress_print = True)
# PRC(NetworkHalfCut, "validation_data", c = "yellow", ls = ":")

# print()

# NetworkOneCut = NNClassifier("minimal_conv2d_cut_1.00VEM/model_converged", supress_print = True)
# PRC(NetworkOneCut, "validation_data", c = "darksalmon", ls = "--")
# PRC(NetworkOneCut, "validation_data_downsampled", c = "darksalmon")

# NetworkOneCut = NNClassifier("minimal_conv2d_1.00VEM_downsampled/model_3", supress_print = True)
# PRC(NetworkOneCut, "validation_data", c = "darksalmon", ls = ":")

# print()

# NetworkTenCut = NNClassifier("minimal_conv2d_cut_10.0VEM/model_converged", supress_print = True)
# PRC(NetworkTenCut, "validation_data", c = "orange", ls = "--")
# PRC(NetworkTenCut, "validation_data_downsampled", c = "orange")

# NetworkTenCut = NNClassifier("minimal_conv2d_3.00VEM_downsampled/model_3", supress_print = True)
# PRC(NetworkTenCut, "validation_data", c = "orange", ls = ":")


NNTest = NNClassifier("minimal_conv2d_real_background/model_converged", "minimal_conv2d_real_background/model_converged", supress_print = True)
NNFiltered = NNClassifier("minimal_conv2d_stations_filtered/model_converged", "minimal_conv2d_stations_filtered/model_converged", supress_print = True)

ROC(NNTest, "validation_data")
ROC(NNFiltered, "validation_data")

print()

# print("\nDATASET".ljust(72) + "TP      FP      TN      FN     sum")
# PRC(CurrentTrigger, "validation_data", c = "k", ls = "--")
# PRC(CurrentTrigger, "validation_data_downsampled", c = "k")
# PRC(NetworkNoCut, "validation_data", c = "green", ls = "--")
# PRC(NetworkNoCut, "validation_data_downsampled", c = "green")
# PRC(NetworkFifthCut, "validation_data", c = "steelblue", ls = "--")
# PRC(NetworkFifthCut, "validation_data_downsampled", c = "steelblue")
# PRC(NetworkHalfCut, "validation_data", c = "yellow", ls = "--")
# PRC(NetworkHalfCut, "validation_data_downsampled", c = "yellow")
# PRC(NetworkOneCut, "validation_data", c = "darksalmon", ls = "--")
# PRC(NetworkOneCut, "validation_data_downsampled", c = "darksalmon")
# PRC(NetworkTenCut, "validation_data", c = "orange", ls = "--")
# PRC(NetworkTenCut, "validation_data_downsampled", c = "orange")

# plt.legend()
plt.show()