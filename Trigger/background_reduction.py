from TriggerStudyBinaries_v2 import *

AllEvents = EventGenerator(["19_19.5"])

# AllEvents.unit_test()

OneLayerEnsemble = Ensemble("one_layer_conv2d")

traces, labels = AllEvents[0].__getitem__(0)

for trace, label in zip(traces, labels):

    print(label, OneLayerEnsemble(trace))
