import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import typing
import os

# custom modules for specific usecase
from TriggerStudyBinaries.Classifier import TriggerClassifier, NNClassifier
from TriggerStudyBinaries.Generator import EventGenerator, Generator
from TriggerStudyBinaries.Signal import VEMTrace, Background

Estimator = typing.Union[TriggerClassifier, NNClassifier]

class TriggerProbabilityDistribution():

    @staticmethod
    # something goes wrong with the first prediction string
    def build_dataset(Classifier : Estimator, Dataset : Generator, save_path : str) -> typing.NoReturn :

        with open(f"/cr/data01/filip/ROC_curves/{save_path}.txt","w") as predictions:
            predictions.write(f"# E/eV sig/VEM SPD theta/° (LBL,n_sig,n_bkg)...\n")

            for batch in range(Dataset.__len__()):
                traces, _ = Dataset.__getitem__(batch, reduce = False)

                print(f"Fetching batch {batch + 1}/{Dataset.__len__()}: {100 * (batch/Dataset.__len__()):.2f}%", end = "...\r")

                for Trace in traces:

                    # ignore mock background traces
                    if Trace.SPDistance == -1:
                        continue

                    save_string = f"{Trace.Energy}\t{Trace.integrate()}\t{Trace.SPDistance}\t{Trace.Zenith} "

                    # gather interesting regions
                    padding = Dataset.window_length + np.random.randint(1,9)
                    interesting_regions = [[Trace._sig_injected_at - padding],[Trace._sig_stopped_at]]

                    try:
                        for start, stop in zip(Trace._bkg_injected_at, Trace._bkg_stopped_at):
                            interesting_regions[0].append(start - padding)
                            interesting_regions[1].append(stop)
                    except AttributeError:
                        pass

                    for start, stop in zip(*interesting_regions):

                        if stop + Dataset.window_length > Trace.trace_length: continue
                        if start < 0: continue

                        for i in range(start, stop, Dataset.window_step):

                            n_sig, n_bkg = Trace.get_n_signal_background_bins(i, Dataset.window_length)
                            pmt_data = Trace.get_trace_window(i, Dataset.window_length, no_label = True)
                            predicted_label = Classifier.model.predict(tf.expand_dims([pmt_data], axis = -1)).argmax()
                            save_string += f"({predicted_label},{n_sig},{n_bkg}) "

                    if save_string != f"{Trace.Energy}\t{Trace.integrate()}\t{Trace.SPDistance}\t{Trace.Zenith} ":
                        with open(f"/cr/data01/filip/ROC_curves/{save_path}.txt","a") as hits:
                            hits.write(save_string+"\n")

    @staticmethod
    def profile_plot(file_path : str) -> typing.NoReturn : 

        def convert_string(string : str) -> list :

            by_column = string.split()
            
            for i, entry in enumerate(by_column):
                if i < 4:
                    by_column[i] = float(entry)
                else:
                    by_column[i] = tuple(map(int, entry[1:-1].split(',')))

            return by_column

        with open(file_path, "r") as input:
            data_by_line = [convert_string(line) for line in input.readlines()[1:]]

# def profile_plot(hits : np.ndarray, misses : np.ndarray, n_bins : int, label : str, color : str) -> typing.NoReturn :

#     minimum_bin = min([hits.min(), misses.min()])
#     maximum_bin = max([hits.max(), misses.max()])
#     bins = np.linspace(minimum_bin, maximum_bin, n_bins)

#     hits_histogrammed, _ = np.histogram(hits, bins = bins)
#     misses_histogrammed, spd = np.histogram(misses, n_bins)

#     for i, (x, o) in enumerate(zip(hits_histogrammed, misses_histogrammed)):

#         if x == 0:
#             continue
#         else:

#             accuracy = x / (x+o)
#             accuracy_err = 1/(x+o)**2 * np.sqrt(x * o**2 + o * x**2 + 2 * x*o)

#             plt.errorbar((spd[:-1] + spd[:1])[i], accuracy, yerr = accuracy_err, capsize = 5, marker = "o", color = color)

#     plt.scatter([],[], c = color, label = label)