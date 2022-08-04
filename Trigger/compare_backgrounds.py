from TriggerStudyBinaries_v2.__configure__ import *
from scipy.stats import norm

plt.rcParams.update({'font.size': 22})

TestRealBackground = EventGenerator("all", real_background = True, split = 1, prior = 0, force_inject = 0)
TestModelBackground = EventGenerator("all", real_background = False, split = 1, prior = 0, force_inject = 0)

c = ["steelblue", "orange"]
l = ["random traces", "model background"]

for i, Dataset in enumerate([TestRealBackground, TestModelBackground]):

    histogram = []

    for batch in range(Dataset.__len__()):

        print(f"Fetching batch {batch + 1}/{Dataset.__len__()}: {100 * (batch/Dataset.__len__()):.2f}%", end = "...\r")

        traces, _ = Dataset.__getitem__(batch)

        for trace in traces:

            histogram.append(np.mean(trace))
    
    # cut real background
    if i == 0: 
        mask = np.where(np.abs(histogram) < 0.01)[0]
        histogram = np.array(histogram)[mask]

    histogram = np.array(histogram) * GLOBAL.ADC_to_VEM
    mu, sigma = np.mean(histogram), np.std(histogram)

    n, bins, _ = plt.hist(histogram, histtype = "step", lw = 2, bins = 100, color = c[i], label = l[i] + f", n = {len(histogram)}", density = True)
    plt.plot(bins, norm.pdf(bins, mu, sigma), ls = "--", lw = 2)

    print(f"{l[i]}: mu = {mu:.3f}, sigma = {sigma:.3f}")

# plt.ylabel("# of occurences")
plt.xlabel("Signal / ADC")
plt.legend()
plt.show()