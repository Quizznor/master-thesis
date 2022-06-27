from binaries.EventGenerators import EventGenerator
from binaries.Signal import VEMTrace

Test = EventGenerator("16_16.5")
Train, Validation = Test()

traces, labels = Train.__getitem__(6)

for trace in traces:
    TestTrace = VEMTrace("SIG", trace = trace)
    TestTrace.plot()

