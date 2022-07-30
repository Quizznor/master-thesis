from TriggerStudyBinaries.Classifier import TriggerClassifier, NNClassifier
from TriggerStudyBinaries.Generator import EventGenerator, Generator
from TriggerStudyBinaries.Signal import VEMTrace, Background

import TriggerStudyBinaries.PerformanceTest as pt
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})

pt.signal_ROC("current_trigger_validation_data_real_background", label = "Current trigger - random traces", ls = "--", c = "k")
pt.signal_ROC("current_trigger_validation_data", label = "Current trigger -  model background", ls = "solid", c = "k")
pt.signal_ROC("mock_linear_validation_data", label = "single layer Conv2d - model background", c = "orange", ls = "solid")
pt.signal_ROC("mock_linear_validation_data_real_background", label = "single layer Conv2d - random traces", c = "orange", ls = "--")
pt.signal_ROC("large_model_validation_data_real_background", label = "large Conv2d - random traces", c = "yellow", ls = "--")
pt.signal_ROC("large_model_validation_data", label = "large Conv2d - model background", c = "yellow", ls = "solid")
pt.signal_ROC("small_model_validation_data_real_background", label = "small Conv2d - random traces", c = "green", ls = "--")
pt.signal_ROC("small_model_validation_data", label = "small Conv2d - model background", c = "green", ls = "solid")
pt.signal_ROC("mock_fifth_validation_data_real_background_uncorrected", label = "single layer Conv2d 0.2 VEM - random traces", c = "darkblue", ls = "--")
pt.signal_ROC("mock_fifth_validation_data_uncorrected", label = "single layer Conv2d 0.2 VEM - model background", c = "darkblue", ls = "solid")
pt.signal_ROC("mock_half_validation_data_real_background_uncorrected", label = "single layer Conv2d 0.5 VEM - random traces", c = "blue", ls = "--")
pt.signal_ROC("mock_half_validation_data_uncorrected", label = "single layer Conv2d 0.5 VEM - model background", c = "blue", ls = "solid")
pt.signal_ROC("mock_one_validation_data_real_background_uncorrected", label = "single layer Conv2d 1.0 VEM - random traces", c = "lightblue", ls = "--")
pt.signal_ROC("mock_one_validation_data_uncorrected", label = "single layer Conv2d 1.0 VEM - model background", c = "lightblue", ls = "solid")

# global plot arguments
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.legend()
plt.show()