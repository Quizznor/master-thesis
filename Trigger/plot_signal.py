from TriggerStudyBinaries.Signal import VEMTrace
import numpy as np
import matplotlib.pyplot as plt

# data = np.loadtxt("/cr/tempdata01/filip/VEM/QGSJET-II/protons/16.5_17/DAT010000.csv")

plt.rcParams.update({'font.size': 22})

# plt.plot(range(len(data[0][4:])), data[0][4:])
test = VEMTrace(n_bins = 20000).Background[0]

ax1 = plt.subplot2grid((1,6),(0,0), colspan = 5)
ax2 = plt.subplot2grid((1,6),(0,5), sharey = ax1)

ax1.axhline(0, zorder = 2, ls = "--", c = "k", lw = 2)
ax1.plot(range(len(test[::20])), test[::20])
ax2.hist(test, orientation = "horizontal", histtype = "step", bins = 20)
ax2.axhline(0, zorder = 2, ls = "--", c = "k", lw = 2)
ax2.axvline(0, 0.1, 0.9, zorder = 2, ls = "--", c = "k", lw = 2)
ax1.axis("off")
ax2.axis("off")
plt.show()