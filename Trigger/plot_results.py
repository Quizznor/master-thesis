from TriggerStudyBinaries.PerformanceTest import TriggerProbabilityDistribution
from TriggerStudyBinaries.Classifier import TriggerClassifier, NNClassifier
from TriggerStudyBinaries.Generator import EventGenerator, Generator
from TriggerStudyBinaries.Signal import VEMTrace, Background
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})

# # signal strength
# TriggerProbabilityDistribution.profile_plot("mock_model_two_layers_2.txt", "signal_strength_sensitivity", label = "CNN - 2 layers, 1 filter", color = "darkgreen")
# TriggerProbabilityDistribution.profile_plot("mock_model_all_two_filters_2.txt", "signal_strength_sensitivity", label = "CNN - 1 layer, 2 filters")
# TriggerProbabilityDistribution.profile_plot("mock_model_all_linear_2.txt", "signal_strength_sensitivity", label = "CNN - 1 layer, 1 filter", color = "orange")
# TriggerProbabilityDistribution.profile_plot("current_triggers_all_traces.txt", "signal_strength_sensitivity", label = "Current triggers", color = "k")

# spd
# TriggerProbabilityDistribution.profile_plot("mock_model_two_layers_2.txt", "SPD_sensitivity", label = "CNN - 2 layers, 1 filter", color = "darkgreen")
# TriggerProbabilityDistribution.profile_plot("mock_model_all_two_filters_2.txt", "SPD_sensitivity", label = "CNN - 1 layer, 2 filters")
# TriggerProbabilityDistribution.profile_plot("mock_model_all_linear_2.txt", "SPD_sensitivity", label = "CNN - 1 layer, 1 filter", color = "orange")
# TriggerProbabilityDistribution.profile_plot("current_triggers_all_traces.txt", "SPD_sensitivity", label = "Current triggers", color = "k")

# TriggerProbabilityDistribution.profile_plot("current_triggers_zero_baseline.txt", "signal_strength_sensitivity", label = "Current triggers, zero baseline", color = "steelblue")
# TriggerProbabilityDistribution.profile_plot("current_triggers_all_traces.txt", "signal_strength_sensitivity", label = "Current triggers", color = "k")


# global plot arguments
plt.xlabel("Shower plane distance / m")
plt.ylabel("Sensitivity")
plt.legend()
plt.show()