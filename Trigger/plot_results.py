from TriggerStudyBinaries_v5 import *

plt.rcParams.update({'font.size': 18})

# pt.ROC("large_model_validation_data", c = "yellow", ls = "solid") # -> trash
# pt.ROC("small_model_validation_data", c = "green", ls = "solid") # -> trash

# # model background
# ROC("current_trigger_validation_data", ls = "solid", c = "k")
# ROC("one_layer_conv2d_cut_0.00VEM_validation_data", c = "orange", ls = "solid")
# # ROC("one_layer_conv2d_cut_0.20VEM_validation_data", c = "darkblue", ls = "solid")
# # ROC("one_layer_conv2d_cut_0.50VEM_validation_data", c = "blue", ls = "solid")
# # ROC("one_layer_conv2d_cut_1.00VEM_validation_data", c = "lightblue", ls = "solid")
# # ROC("one_layer_conv2d_cut_2.00VEM_validation_data", c = "yellow", ls = "solid")
# # ROC("one_layer_conv2d_cut_4.63VEM_validation_data", c = "green", ls = "solid")
# ROC("one_layer_conv2d_cut_10.0VEM_validation_data", c = "brown", ls = "solid")
# # ROC("bayes_m249_validation_data", c = "darksalmon", ls = "solid")
# # ROC("bayes_m244_validation_data", c = "lime", ls = "solid")
# # ROC("bayes_LHQ_validation_data", c = "steelblue", ls = "solid")

# print()

# # model background
# ROC("current_trigger_random_traces", ls = "--", c = "k")
# ROC("one_layer_conv2d_cut_0.00VEM_random_traces", c = "orange", ls = "--")
# # ROC("one_layer_conv2d_cut_0.20VEM_random_traces", c = "darkblue", ls = "--")
# # ROC("one_layer_conv2d_cut_0.50VEM_random_traces", c = "blue", ls = "--")
# # ROC("one_layer_conv2d_cut_1.00VEM_random_traces", c = "lightblue", ls = "--")
# # ROC("one_layer_conv2d_cut_2.00VEM_random_traces", c = "yellow", ls = "--")
# # ROC("one_layer_conv2d_cut_4.63VEM_random_traces", c = "green", ls = "--")
# ROC("one_layer_conv2d_cut_10.0VEM_random_traces", c = "brown", ls = "--")
# # ROC("bayes_m249_random_traces", c = "darksalmon", ls = "--")
# # ROC("bayes_m244_random_traces", c = "lime", ls = "--")
# # ROC("bayes_LHQ_random_traces", c = "steelblue", ls = "--")

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
NetworkNoCut = NNClassifier("minimal_conv2d_cut_0.00VEM/model_converged", supress_print = True)
NetworkFifthCut = NNClassifier("minimal_conv2d_cut_0.20VEM/model_converged", supress_print = True)
NetworkHalfCut = NNClassifier("minimal_conv2d_cut_0.50VEM/model_converged", supress_print = True)
NetworkOneCut = NNClassifier("minimal_conv2d_cut_1.00VEM/model_converged", supress_print = True)
NetworkTenCut = NNClassifier("minimal_conv2d_cut_10.0VEM/model_converged", supress_print = True)


print("\nDATASET".ljust(72) + "TP      FP      TN      FN     sum")
ROC(CurrentTrigger, "validation_data", c = "k", ls = "--")
ROC(CurrentTrigger, "validation_data_downsampled", c = "k")
ROC(NetworkNoCut, "validation_data", c = "green", ls = "--")
ROC(NetworkNoCut, "validation_data_downsampled", c = "green")
ROC(NetworkFifthCut, "validation_data", c = "steelblue", ls = "--")
ROC(NetworkFifthCut, "validation_data_downsampled", c = "steelblue")
ROC(NetworkHalfCut, "validation_data", c = "yellow", ls = "--")
ROC(NetworkHalfCut, "validation_data_downsampled", c = "yellow")
ROC(NetworkOneCut, "validation_data", c = "darksalmon", ls = "--")
ROC(NetworkOneCut, "validation_data_downsampled", c = "darksalmon")
ROC(NetworkTenCut, "validation_data", c = "orange", ls = "--")
ROC(NetworkTenCut, "validation_data_downsampled", c = "orange")

plt.legend()
plt.show()

# model background
ROC("current_trigger_validation_data", ls = "solid", c = "k")
ROC("one_layer_conv2d_cut_0.00VEM_validation_data", c = "orange", ls = "solid")
# ROC("one_layer_conv2d_cut_0.20VEM_validation_data", c = "darkblue", ls = "solid")
# ROC("one_layer_conv2d_cut_0.50VEM_validation_data", c = "blue", ls = "solid")
# ROC("one_layer_conv2d_cut_1.00VEM_validation_data", c = "lightblue", ls = "solid")
# ROC("one_layer_conv2d_cut_2.00VEM_validation_data", c = "yellow", ls = "solid")
ROC("one_layer_conv2d_cut_4.63VEM_validation_data", c = "green", ls = "solid")
# ROC("bayes_m249_validation_data", c = "darksalmon", ls = "solid")
# ROC("bayes_m244_validation_data", c = "lime", ls = "solid")
ROC("bayes_LHQ_validation_data", c = "steelblue", ls = "solid")

print()

# model background
ROC("current_trigger_random_traces", ls = "--", c = "k")
ROC("one_layer_conv2d_cut_0.00VEM_random_traces", c = "orange", ls = "--")
# ROC("one_layer_conv2d_cut_0.20VEM_random_traces", c = "darkblue", ls = "--")
# ROC("one_layer_conv2d_cut_0.50VEM_random_traces", c = "blue", ls = "--")
# ROC("one_layer_conv2d_cut_1.00VEM_random_traces", c = "lightblue", ls = "--")
# ROC("one_layer_conv2d_cut_2.00VEM_random_traces", c = "yellow", ls = "--")
ROC("one_layer_conv2d_cut_4.63VEM_random_traces", c = "green", ls = "--")
# ROC("bayes_m249_random_traces", c = "darksalmon", ls = "--")
# ROC("bayes_m244_random_traces", c = "lime", ls = "--")
ROC("bayes_LHQ_random_traces", c = "steelblue", ls = "--")

plt.legend(ncol = 2)
plt.show()

