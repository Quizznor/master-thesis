from TriggerStudyBinaries_v4 import *

plt.rcParams.update({'font.size': 18})

print("\nDATASET".ljust(62) + "TP      FP      TN      FN      sum")

# model background
ROC("current_trigger_validation_data", ls = "solid", c = "k")
ROC("one_layer_conv2d_cut_0.00VEM_validation_data", c = "orange", ls = "solid")
# pt.ROC("large_model_validation_data", c = "yellow", ls = "solid") # -> trash
# pt.ROC("small_model_validation_data", c = "green", ls = "solid") # -> trash
ROC("one_layer_conv2d_cut_0.20VEM_validation_data", c = "darkblue", ls = "solid")
ROC("one_layer_conv2d_cut_0.50VEM_validation_data", c = "blue", ls = "solid")
ROC("one_layer_conv2d_cut_1.00VEM_validation_data", c = "lightblue", ls = "solid")
ROC("one_layer_conv2d_cut_2.00VEM_validation_data", c = "yellow", ls = "solid")
ROC("one_layer_conv2d_cut_4.63VEM_validation_data", c = "green", ls = "solid")

print()

# model background
ROC("current_trigger_random_traces", ls = "--", c = "k")
ROC("one_layer_conv2d_cut_0.00VEM_random_traces", c = "orange", ls = "--")
ROC("one_layer_conv2d_cut_0.20VEM_random_traces", c = "darkblue", ls = "--")
ROC("one_layer_conv2d_cut_0.50VEM_random_traces", c = "blue", ls = "--")
ROC("one_layer_conv2d_cut_1.00VEM_random_traces", c = "lightblue", ls = "--")
ROC("one_layer_conv2d_cut_2.00VEM_random_traces", c = "yellow", ls = "--")
ROC("one_layer_conv2d_cut_4.63VEM_random_traces", c = "green", ls = "--")


plt.legend(ncol = 2)
plt.show()