import matplotlib.pyplot as plt
import numpy as np
import sys, os

files = os.listdir(os.environ["DATA"] + "01_simulation/component_signal/")

for file in files:
    data = np.loadtxt(os.environ["DATA"] + "01_simulation/component_signal/" + file)

    for i, trace in enumerate(data):

        if np.max(data) in trace:
            plt.plot(range(len(trace)), trace, label = i)
        else:
            plt.plot(range(len(trace)), trace)

    plt.legend()
    plt.show()