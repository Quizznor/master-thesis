#!/usr/bin/python3

import sys

sys.dont_write_bytecode = True

from binaries import *
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

mock_baseline = np.random.normal(np.random.uniform(low = - 0.5 / 61.75, high = 0.5 / 61.75), 0.5 / 61.75, 20000)
mock_signal = np.random.choice(os.listdir("/cr/data01/filip/first_simulation/tensorflow/signal/"))
pmts = np.loadtxt(f"/cr/data01/filip/first_simulation/tensorflow/signal/{mock_signal}")
mock_measured = mock_baseline
start = np.random.randint(-20000, -2048)
mock_measured[start : start + 2048] += pmts[0]

trace = mock_measured

plt.plot(range(len(trace)), trace, c = "steelblue")
plt.yscale("log")
plt.xlim(0, 20000)
plt.ylim(5 * 10e-4, 0.7 * 10e2)
plt.show()

# Dataset = DataSetGenerator("second_simulation/tensorflow/signal/", train = False)
# traces , labels = Dataset.__getitem__(0)
# t = np.arange(0,20000, 1)

# signal = traces[np.random.choice(np.asarray(labels[:,0] == 0).nonzero()[0])]
# background = traces[np.random.choice(np.asarray(labels[:,0] == 1).nonzero()[0])]

# fig, axes = plt.subplots(3, sharex = True)

# for i,ax in enumerate(axes,1):
#     ax.set_title(f"PMT #{i}")
#     ax.set_yscale('log')
#     ax.set_xlim(0, 20000)
#     ax.set_ylim(5 * 10e-4, 0.7 * 10e2)
#     ax.set_yticks([10e-2, 10e-1, 10e0, 10e1])

# axes[2].set_xlabel("Time bin (8.3 ns)")
# fig.text(0.07, 0.5, 'PMT signal strength (VEM)', va='center', rotation='vertical')

# for i, trace in enumerate((background, signal)):

#     color = "r" if i == 1 else "b"
#     pmt1, pmt2, pmt3 = trace[0:20000], trace[20000:40000], trace[40000:60000]
#     axes[0].plot(t, pmt1, c = color, alpha = 0.8), axes[1].plot(t,pmt2, c = color, alpha = 0.8), axes[2].plot(t,pmt3, c = color, alpha = 0.8)

# plt.show()