# builtin modules
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 22})

# custom modules
from binaries.Classifiers import NNClassifier, TriggerClassifier
from binaries.EventGenerators import EventGenerator
from binaries.PerformanceTest import trigger_probability_distribution, profile_plot

# Classifier = TriggerClassifier()
# hits, misses = trigger_probability_distribution(Classifier, "all", split = 0.8)
# np.savetxt(f"/cr/data01/filip/trigger_probabilities/all_hits_old_triggers.txt", np.array(hits))
# np.savetxt(f"/cr/data01/filip/trigger_probabilities/all_misses_old_triggers.txt", np.array(misses))

# Classifier = NNClassifier("/cr/data01/filip/all_traces_model_high_noise/model_2")
hits_NN, misses_NN = [], []

colors = ["steelblue", "orange", "green", "red", "brown", "k", "lime"]
# for i, sets in enumerate(["16_16.5", "16.5_17","17_17.5", "17.5_18", "18_18.5", "18.5_19", "19_19.5"]):
for i, sets in enumerate(["16.5_17","17_17.5", "17.5_18", "18_18.5", "18.5_19", "19_19.5"]):
# for sets in ["16_16.5"]:

    # Dataset = EventGenerator(sets, split = 1, prior = 1, sigma = 2, mu = [-2, 2])

# #     # create datasets
#     hits, misses = trigger_probability_distribution(Classifier, Dataset)
#     np.savetxt(f"/cr/data01/filip/trigger_probabilities/all_traces_model_2/all_hits_{sets}_high_noise.txt", np.array(hits))
#     np.savetxt(f"/cr/data01/filip/trigger_probabilities/all_traces_model_2/all_misses_{sets}_high_noise.txt", np.array(misses))

    # load and show datasets
    _, _, hits, _ = np.loadtxt(f"/cr/data01/filip/trigger_probabilities/all_traces_model_2/val_hits_{sets}_high_noise.txt", unpack = True)
    _, _, misses, _ = np.loadtxt(f"/cr/data01/filip/trigger_probabilities/all_traces_model_2/val_misses_{sets}_high_noise.txt", unpack = True)#
    # profile_plot(hits, misses, 38, sets, color = colors[i])
    hits_NN.append(hits), misses_NN.append(misses)

_, _, hits_OLD, _ = np.loadtxt("/cr/data01/filip/trigger_probabilities/old_triggers_hits_high_noise.txt", unpack = True)
_, _, misses_OLD, _ = np.loadtxt("/cr/data01/filip/trigger_probabilities/old_triggers_misses_high_noise.txt", unpack = True)
profile_plot(np.concatenate(hits_NN), np.concatenate(misses_NN), 55, "CNN triggers", color = "steelblue")
profile_plot(hits_OLD, misses_OLD, 55, "Th + ToT", color = "orange")

# # Classifier = TriggerClassifier()
# # Dataset = EventGenerator("all", split = 1, prior = 1, sigma = 2, mu = [-2, 2])
# # hits, misses = trigger_probability_distribution(Classifier, Dataset)
# # np.savetxt(f"/cr/data01/filip/trigger_probabilities/old_triggers_hits_high_noise.txt", np.array(hits))
# # np.savetxt(f"/cr/data01/filip/trigger_probabilities/old_triggers_misses_high_noise.txt", np.array(misses))

plt.title("Lateral trigger probability")
plt.xlabel("Shower plane distance / m")
plt.ylabel("TP / (TP + FP)")
# plt.ylim(0.8, 1.05)
plt.legend()
plt.show()
