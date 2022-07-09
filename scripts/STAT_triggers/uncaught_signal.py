# from binaries.EventGenerators import EventGenerator
from binaries.Signal import VEMTrace
import matplotlib.pyplot as plt
import numpy as np

# LowSignals = EventGenerator("17_17.5", split = 1, prior = 1, sigma = 2, mu = [-2, 2])
# Network = NNClassifier("/cr/data01/filip/all_traces_model_high_noise/model_2")
# Current = TriggerClassifier()

trace_data = np.loadtxt("/cr/tempdata01/filip/QGSJET-II/protons/17_17.5/DAT080911.csv")
trace_data =  np.split(trace_data, len(trace_data) / 3 )
np.random.seed(0)

Trace = VEMTrace(trace_data[-1], sigma = 2, mu = [-2, 2])

plt.rcParams.update({'font.size': 22})
ax1 = plt.gca()

for i, trace in enumerate(Trace(pooling = False), 1):
    trace = trace[18250:18600]
    ax1.plot(range(len(trace)), trace, label = f"PMT #{i}")

plt.text(81, 0.14, "120 bin sliding window", fontsize = 18)
plt.fill_between(range(40, 160), [0 for i in range(120)], [0.2 for i in range(120)], color = "r", alpha = 0.3)
plt.xlabel("Time bin / 8.3 ns")
plt.ylabel("Signal / VEM")
plt.legend()
plt.show()





