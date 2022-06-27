from unittest import skip
from binaries.Classifiers import TriggerClassifier, NNClassifier
from binaries.EventGenerators import EventGenerator
from binaries.Signal import VEMTrace
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import typing

def lateral_trigger_probability_distribution(Classifier : typing.Union[TriggerClassifier, NNClassifier], libraries : typing.Union[list, str], split : float = 1) -> tuple :

    if split != 1:
        _, Dataset = EventGenerator(libraries, split = split, prior = 1)
    else:
        Dataset = EventGenerator(libraries, split = 1, prior = 1)
    hits, misses = [], []

    for batch in range(Dataset.__len__()):
        traces, labels = Dataset.__getitem__(batch, for_training = False)

        for Trace, _ in zip(traces, labels):

            if Trace.SPDistance == -1:
                continue

            predicted_label = Classifier(Trace)

            if predicted_label:
                hits.append(Trace.SPDistance)
            else:
                misses.append(Trace.SPDistance)

        print(f"Fetching VEM Traces: {100 * (batch/Dataset.__len__()):.2f}%", end = "...\r")
    
    return hits, misses

def lateral_trigger_profile_plot(hits : np.ndarray, misses : np.ndarray, n_bins : int, label : str, color : str) -> typing.NoReturn : 

    hits_histogrammed, _ = np.histogram(hits, n_bins, (0, 4000))
    misses_histogrammed, spd = np.histogram(misses, n_bins, (0, 4000))
    last_drawn = None

    # # for the LDF
    # beta, gamma = 18, 2
    # spd = np.power(spd / 1000, beta) * np.power((spd + 700) / (1700), beta + gamma)

    for i, (x, o) in enumerate(zip(hits_histogrammed, misses_histogrammed)):

        if x == 0 or o == 0:
            last_drawn = None
            continue
        else:
            plt.plot([spd[i], spd[i+1]], [x / (x + o) for j in range(2)], c = color)    # add the probability
            if last_drawn is not None:                           
                plt.plot([spd[i],spd[i]], [last_drawn , x / (x+o)], c = color)          # add connecting steps

            last_drawn = x / (x + o)

    plt.plot([],[], c = color, label = label)