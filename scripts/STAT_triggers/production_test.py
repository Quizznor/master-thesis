import numpy as np

# from binaries.EventGenerators import EventGenerator
from binaries.Signal import VEMTrace

# Test = EventGenerator("16_16.5")
# Train, Validation = Test()

# traces, labels = Train.__getitem__(6)

# for trace in traces:
#     TestTrace = VEMTrace("SIG", trace = trace)
#     TestTrace.plot()

mock_signal = np.array([np.linspace(0,0.5,5) for i in range(3)])

Test = VEMTrace(mock_signal)
Test.plot()

# import numpy as np
# import matplotlib.pyplot as plt

# data = np.loadtxt("/cr/tempdata01/filip/background/uub.dat")
# print(data.shape)

# while True:
#     i = np.random.randint(0,len(data))

#     plt.plot(range(len(data[i])), data[i])
#     plt.show()