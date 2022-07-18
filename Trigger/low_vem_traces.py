from TriggerStudyBinaries.Classifier import TriggerClassifier, NNClassifier
from TriggerStudyBinaries.Generator import EventGenerator, Generator
from TriggerStudyBinaries.Signal import VEMTrace, Background

import sys

if sys.argv[1] == "build":

    Dataset = EventGenerator("all", prior = 1, split = 1, seed = True, mu = 0, sigma = 0)
    Trigger = TriggerClassifier()

    for batch in range(Dataset.__len__()):

        print(f"Fetching batch {batch + 1}/{Dataset.__len__()}", end = "\r")
        traces, labels = Dataset.__getitem__(batch)

        for trace in traces:
            if VEMTrace.integrate(trace) < 1.75:

                if Trigger(trace):

                    print("Found a vem trace")

                    with open("/cr/tempdata01/filip/00_MISC/low_vem_traces.txt","a") as file:
                        file.savetxt(trace)