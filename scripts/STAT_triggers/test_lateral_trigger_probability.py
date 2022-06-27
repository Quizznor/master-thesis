# builtin modules
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 22})

# custom modules
from binaries.Classifiers import NNClassifier, TriggerClassifier
from binaries.EventGenerators import EventGenerator
from binaries.PerformanceTest import lateral_trigger_probability_distribution, lateral_trigger_profile_plot

Classifier = NNClassifier("/cr/data01/filip/all_traces_model/model_1")
# Classifier = TriggerClassifier()

colors = ["steelblue", "orange", "green", "red", "brown"]
for i, sets in enumerate(["17_17.5", "17.5_18", "18_18.5", "18.5_19", "19_19.5"]):

#     # # create datasets
#     # hits, misses = lateral_trigger_probability_distribution(Classifier, sets, split = 0.8)
#     # np.savetxt(f"/cr/data01/filip/lateral_trigger_probability/all_traces_model_1/val_hits_{sets}.txt", np.array(hits))
#     # np.savetxt(f"/cr/data01/filip/lateral_trigger_probability/all_traces_model_1/val_misses_{sets}.txt", np.array(misses))

    # load and show datasets
    hits_NN = np.loadtxt(f"/cr/data01/filip/lateral_trigger_probability/all_traces_model_1/val_hits_{sets}.txt")
    misses_NN = np.loadtxt(f"/cr/data01/filip/lateral_trigger_probability/all_traces_model_1/val_misses_{sets}.txt")
    lateral_trigger_profile_plot(hits_NN, misses_NN, 28, f"log E {sets}", colors[i])

# hits_OLD = np.loadtxt("/cr/data01/filip/lateral_trigger_probability/all_traces_old_triggers_hits.txt")
# misses_OLD = np.loadtxt("/cr/data01/filip/lateral_trigger_probability/all_traces_old_triggers_misses.txt")
# lateral_trigger_profile_plot(hits_OLD, misses_OLD, 38, "Current triggers", color = "orange")

# hits_NN = np.loadtxt("/cr/data01/filip/lateral_trigger_probability/all_traces_model_1/all_val_hits.txt")
# misses_NN = np.loadtxt("/cr/data01/filip/lateral_trigger_probability/all_traces_model_1/all_val_misses.txt")
# lateral_trigger_profile_plot(hits_NN, misses_NN, 38, "CNN triggers", color = "steelblue")

# plt.title("Lateral trigger probability by energy")
plt.xlabel("Shower plane distance / m")
plt.ylabel("Trigger probability")
plt.ylim(0.8, 1.05)
plt.legend()
plt.show()
