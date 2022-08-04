from TriggerStudyBinaries_v2.__configure__ import *

print("\nDATASET".ljust(62) + "TP      FP      TN      FN      sum")

# model background
pt.ROC("current_trigger_validation_data", label = "Current trigger -  model background", ls = "solid", c = "k")
pt.ROC("mock_linear_validation_data", label = "single layer Conv2d - model background", c = "orange", ls = "solid")
pt.ROC("large_model_validation_data", label = "large Conv2d - model background", c = "yellow", ls = "solid")
pt.ROC("small_model_validation_data", label = "small Conv2d - model background", c = "green", ls = "solid")
pt.ROC("mock_fifth_validation_data_uncorrected", label = "single layer Conv2d 0.2 VEM - model background", c = "darkblue", ls = "solid")
pt.ROC("mock_half_validation_data_uncorrected", label = "single layer Conv2d 0.5 VEM - model background", c = "blue", ls = "solid")
pt.ROC("mock_one_validation_data_uncorrected", label = "single layer Conv2d 1.0 VEM - model background", c = "lightblue", ls = "solid")

print()

# real background
pt.ROC("current_trigger_validation_data_real_background", label = "Current trigger - random traces", ls = "--", c = "k")
pt.ROC("mock_linear_validation_data_real_background", label = "single layer Conv2d - random traces", c = "orange", ls = "--")
pt.ROC("large_model_validation_data_real_background", label = "large Conv2d - random traces", c = "yellow", ls = "--")
pt.ROC("small_model_validation_data_real_background", label = "small Conv2d - random traces", c = "green", ls = "--")
pt.ROC("mock_fifth_validation_data_real_background_uncorrected", label = "single layer Conv2d 0.2 VEM - random traces", c = "darkblue", ls = "--")
# pt.ROC("mock_half_validation_data_real_background_uncorrected", label = "single layer Conv2d 0.5 VEM - random traces", c = "blue", ls = "--")
# pt.ROC("mock_one_validation_data_real_background_uncorrected", label = "single layer Conv2d 1.0 VEM - random traces", c = "lightblue", ls = "--")

plt.xlabel("False positive rate")
plt.ylabel("True positive rate")

plt.figure()
# model background
pt.PRC("current_trigger_validation_data", label = "Current trigger -  model background", ls = "solid", c = "k")
pt.PRC("mock_linear_validation_data", label = "single layer Conv2d - model background", c = "orange", ls = "solid")
pt.PRC("large_model_validation_data", label = "large Conv2d - model background", c = "yellow", ls = "solid")
pt.PRC("small_model_validation_data", label = "small Conv2d - model background", c = "green", ls = "solid")
pt.PRC("mock_fifth_validation_data_uncorrected", label = "single layer Conv2d 0.2 VEM - model background", c = "darkblue", ls = "solid")
pt.PRC("mock_half_validation_data_uncorrected", label = "single layer Conv2d 0.5 VEM - model background", c = "blue", ls = "solid")
pt.PRC("mock_one_validation_data_uncorrected", label = "single layer Conv2d 1.0 VEM - model background", c = "lightblue", ls = "solid")


print()

# real background
pt.PRC("current_trigger_validation_data_real_background", label = "Current trigger - random traces", ls = "--", c = "k")
pt.PRC("mock_linear_validation_data_real_background", label = "single layer Conv2d - random traces", c = "orange", ls = "--")
pt.PRC("large_model_validation_data_real_background", label = "large Conv2d - random traces", c = "yellow", ls = "--")
pt.PRC("small_model_validation_data_real_background", label = "small Conv2d - random traces", c = "green", ls = "--")
pt.PRC("mock_fifth_validation_data_real_background_uncorrected", label = "single layer Conv2d 0.2 VEM - random traces", c = "darkblue", ls = "--")
pt.PRC("mock_half_validation_data_real_background_uncorrected", label = "single layer Conv2d 0.5 VEM - random traces", c = "blue", ls = "--")
pt.PRC("mock_one_validation_data_real_background_uncorrected", label = "single layer Conv2d 1.0 VEM - random traces", c = "lightblue", ls = "--")

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.legend()
plt.show()