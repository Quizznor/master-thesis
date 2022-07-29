from TriggerStudyBinaries.Classifier import TriggerClassifier, NNClassifier
from TriggerStudyBinaries.Signal import VEMTrace, Background, Baseline
from TriggerStudyBinaries.Generator import EventGenerator, Generator

import matplotlib.pyplot as plt
import numpy as np

SampleModel = EventGenerator(["16_16.5"], prior = 0, split = 1, real_background = False)
SampleReal = EventGenerator(["16_16.5"], prior = 0, split = 1, real_background = True)

for Dataset in [SampleModel, SampleReal]:

    for batch in range(Dataset.__len__()):

        traces, _ = Dataset.__getitem__(batch, reduce = False)

        for Trace in traces:

            plt.hist(np.mean(Trace.Baseline, axis = 2), histtype = "step")
            
            break
        break
    

plt.show()