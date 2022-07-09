from binaries.Classifiers import TriggerClassifier, NNClassifier
from binaries.EventGenerators import Generator
from binaries.Signal import VEMTrace
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import typing

def trigger_probability_distribution(Classifier : typing.Union[TriggerClassifier, NNClassifier], Dataset : Generator) -> tuple :

    hits, misses = [], []

    for batch in range(Dataset.__len__()):
        traces, labels = Dataset.__getitem__(batch, for_training = False)

        for Trace, _ in zip(traces, labels):

            if Trace.is_background:
                continue

            predicted_label = Classifier(Trace)

            if predicted_label:
                hits.append([Trace.integrate(), Trace.Energy, Trace.SPDistance, Trace.Zenith])
            else:
                misses.append([Trace.integrate(), Trace.Energy, Trace.SPDistance, Trace.Zenith])

        print(f"Fetching VEM Traces: {100 * (batch/Dataset.__len__()):.2f}%", end = "...\r")
    
    print(f"Library done...", end = "\r\n")

    return hits, misses

def profile_plot(hits : np.ndarray, misses : np.ndarray, n_bins : int, label : str, color : str) -> typing.NoReturn :

    minimum_bin = min([hits.min(), misses.min()])
    maximum_bin = max([hits.max(), misses.max()])
    bins = np.linspace(minimum_bin, maximum_bin, n_bins)

    hits_histogrammed, _ = np.histogram(hits, bins = bins)
    misses_histogrammed, spd = np.histogram(misses, n_bins)

    for i, (x, o) in enumerate(zip(hits_histogrammed, misses_histogrammed)):

        if x == 0:
            continue
        else:

            accuracy = x / (x+o)
            accuracy_err = 1/(x+o)**2 * np.sqrt(x * o**2 + o * x**2 + 2 * x*o)

            plt.errorbar((spd[:-1] + spd[:1])[i], accuracy, yerr = accuracy_err, capsize = 5, marker = "o", color = color)

    plt.scatter([],[], c = color, label = label)