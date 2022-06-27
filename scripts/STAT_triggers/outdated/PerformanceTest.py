from binaries.Classifiers import TriggerClassifier, NNClassifier
from binaries.EventGenerators import Generator
from binaries.Signal import VEMTrace
import tensorflow as tf
import numpy as np
import typing

def lateral_trigger_probability_distribution(Classifier : typing.Any, Dataset : Generator) -> tuple :

    hits, misses = [], []

    # for batch in range(Set.__len__()):
    for batch in range(4):
        traces, labels = Dataset.__getitem__(batch, for_training = False)

        for trace, label in zip(traces, labels):

            Trace = VEMTrace("SIG", trace = trace)
            predicted_label = Classifier(Trace)

            if predicted_label:
                hits.append(Trace._SPDistance)
            else:
                misses.append(Trace._SPDistance)

        print(f"{50 * (batch/Dataset.__len__()):.2f}%", end = "")
    
    return hits, misses