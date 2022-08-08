from TriggerStudyBinaries_v2 import *

plt.rcParams.update({'font.size': 22})

TestRealBackground = EventGenerator("all", real_background = True, split = 1, prior = 0, ADC_to_VEM = 1, force_inject = 0)
TestModelBackground = EventGenerator("all", real_background = False, split = 1, prior = 0, ADC_to_VEM = 1)

c = ["steelblue", "orange"]
l = ["random traces", "model background"]

fig, (ax1, ax2) = plt.subplots(2)

for i, Dataset in enumerate([TestRealBackground, TestModelBackground]):

    histo_mean, difference_mean = [], []

    for batch in range(Dataset.__len__()):

        print(f"Fetching batch {batch + 1}/{Dataset.__len__()}: {100 * (batch/Dataset.__len__()):.2f}%", end = "...\r")

        traces, _ = Dataset.__getitem__(batch)

        for trace in traces:

            diff = np.diff(trace)

            histo_mean.append(np.mean(trace))
            difference_mean.append(np.mean(diff))

    n, bins, _ = ax1.hist(histo_mean, histtype = "step", lw = 2, bins = 100, range = (-1,1), color = c[i], label = l[i] + f", n = {len(histo_mean)}", density = True)
    n, bins, _ = ax2.hist(difference_mean, histtype = "step", lw = 2, bins = 30, range = (-0.05,0.05), color = c[i], label = l[i] + f", n = {len(difference_mean)}", density = True)

    print(f"{l[i]}: mu = {np.mean(histo_mean):.3f}, diff = {np.mean(difference_mean):.6f}")

ax1.set_ylabel("Histogram")
ax2.set_ylabel("Difference")
ax1.set_xlabel("Signal / ADC")
ax2.set_xlabel("Signal / ADC")

plt.show()