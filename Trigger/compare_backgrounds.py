from TriggerStudyBinaries_v2.__configure__ import *

Data, _ = EventGenerator("all", force_inject = 3, real_background = True, seed = 29384)

Data.__getitem__(0)

# import matplotlib.pyplot as plt
# import numpy as np
# import scipy.stats

# SampleModel = EventGenerator(["19_19.5"], prior = 0, split = 1, real_background = False)
# SampleReal = EventGenerator(["19_19.5"], prior = 0, split = 1, real_background = True)
# fig, ax = plt.subplots(3,1, sharex = True)
# [axis.set_ylabel(f"PMT #{i + 1}") for i, axis in enumerate(ax)]
# ax[-1].set_xlabel("FADC count")

# colors = ["steelblue", "orange"]
# label = ["Model background", "Random traces"]

# for j, Dataset in enumerate([SampleModel, SampleReal]):

#     if j == 0: continue
#     bin_contents = [[] for i in range(3)]

#     for batch in range(Dataset.__len__()):

#         print(f"Fetching batch {batch + 1}/{Dataset.__len__()}: {100 * (batch/Dataset.__len__()):.2f}%", end = "...\r")

#         traces, _ = Dataset.__getitem__(batch, reduce = False)

#         for Trace in traces:
#             for i, pmt in enumerate(Trace.Baseline):
#                 for bin in pmt: bin_contents[i].append(bin)

#     for i in range(3):

#         mu, sigma =  np.mean(bin_contents[i]), np.std(bin_contents[i])
#         ax[i].axvline(mu, c = colors[j], ls = "--")
#         bins, edges = np.histogram(bin_contents[i], bins = 100)
#         best_fit_line = scipy.stats.norm.pdf(edges, mu, sigma)
#         ax[i].plot(edges[:-1], bins, lw = 2, color = colors[j], label = label[j])

#         print(f"{label[j]} - PMT #{i + 1}: mu = {mu:.5f}, sigma = {sigma:.5f}")



# ax[-1].legend()
# plt.show()