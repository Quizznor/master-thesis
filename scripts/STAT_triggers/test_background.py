# builtin modules
import numpy as np

# custom modules
from binaries.Classifiers import NNClassifier, TriggerClassifier
from binaries.EventGenerators import EventGenerator
from binaries.Signal import VEMTrace

BackgroundData = EventGenerator("all", prior = 0, split = 1, sigma = 2, mu = [-2, 2])
# Classifier = NNClassifier("/cr/data01/filip/all_traces_model_high_noise/model_2")
Classifier = TriggerClassifier()
False_Negatives, True_Negatives = 0, 0

for batch in range(1):
        traces, labels = BackgroundData.__getitem__(batch, for_training = False)
        print(f"Evaluating batch {batch}/{BackgroundData.__len__()}...", end = "\r")

        for Trace, _ in zip(traces, labels):

            if Classifier(Trace):
                False_Negatives += 1
            else: True_Negatives += 1

# print(False_Negatives, True_Negatives, "                        ")
# outputs 0 66402 <- absolutely no false negatives! :o
# outputs 38650 27752 at mean and std = 2 ADC counts
# outputs 0 81864 at mean and std = 2 ADC counts for all_traces_model_high_noise
# outputs         at mean and std = 2 ADC counts for classical triggers
