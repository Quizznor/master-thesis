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
    def build_dataset(Classifier : Estimator, Dataset : Generator, save_path : str) -> typing.NoReturn :

        os.makedirs(f"/cr/data01/filip/ROC_curves/{save_path}", exist_ok = True)

        with open(f"/cr/data01/filip/ROC_curves/{save_path}/hits.txt","w") as hits:
            hits.write(f"LBL\tn_sig\tn_bkg\tE (eV)\tsig (VEM)\tSPD\ttheta (°)\n")

        with open(f"/cr/data01/filip/ROC_curves/{save_path}/misses.txt","w") as misses:
            misses.write(f"LBL\tn_sig\tn_bkg\tE (eV)\tsig (VEM)\tSPD\ttheta (°)\n")

        for batch in range(Dataset.__len__()):
            traces, _ = Dataset.__getitem__(batch, reduce = False)

            print(f"Fetching batch {batch + 1}/{Dataset.__len__()}: {100 * (batch/Dataset.__len__()):.2f}%", end = "...\r")

            for Trace in traces:

                # ignore mock background traces
                if Trace.SPDistance == -1:
                    continue

                for i in range(0, Trace.trace_length - EventGenerator.Window, EventGenerator.Step):
                    
                    n_sig, n_bkg = Trace.get_n_signal_background_bins(i, EventGenerator.Window)

                    # continue only if trace does not (only) contain baseline
                    if n_sig == n_bkg == 0:
                        continue

                    label, pmt_data = Trace.get_trace_window(i, EventGenerator.Window)
                    predicted_label = Classifier.model.predict(tf.expand_dims([pmt_data], axis = -1)).argmax()
                    save_string = f"{label}\t{n_sig}\t{n_bkg}\t{Trace.Energy}\t{Trace.integrate()}\t{Trace.SPDistance}\t{Trace.Zenith}\n"

                    if predicted_label == label:
                        with open(f"/cr/data01/filip/ROC_curves/{save_path}/hits.txt","a") as hits:
                            hits.write(save_string)
                    else:
                        with open(f"/cr/data01/filip/ROC_curves/{save_path}/misses.txt","a") as misses:
                            misses.write(save_string)

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