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
    # something goes wrong with the first prediction string for some reason
    def build_dataset(Classifier : Estimator, Dataset : Generator, save_path : str) -> typing.NoReturn :

        for batch in range(Dataset.__len__()):
            traces, _ = Dataset.__getitem__(batch, reduce = False)

            print(f"Fetching batch {batch + 1}/{Dataset.__len__()}: {100 * (batch/Dataset.__len__()):.2f}%", end = "...\r")

            for Trace in traces:

                # ignore mock background traces
                if Trace.SPDistance == -1:
                    continue

                save_string = f"{Trace.Energy:.3e} {Trace.SPDistance:.0f} {Trace.Zenith:.3f} "

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
                        predicted_label = Classifier(pmt_data)

                        save_string += f"({predicted_label},{n_sig},{n_bkg},{VEMTrace.integrate(pmt_data)}) "

                if len(save_string) < 25:
                    continue

                with open(f"/cr/data01/filip/ROC_curves/{save_path}","a") as hits:
                    hits.write(save_string+"\n")

    @staticmethod
    def profile_plot(file_path : str, task : str, **kwargs) -> typing.NoReturn : 

        class Prediction():

            def __init__(self, *args) -> typing.NoReturn : 

                self.Energy = float(args[0])        # energy of (to be) predicted shower
                self.SPD    = int(args[1])          # shower plane distance of shower
                self.Zenith = float(args[2])        # zenith of the primary particle

                self.Predictions = []               # the individual predictions

                for arg in args[3:]:
                    lbl, n_sig, n_bkg, sig = arg[1:-1].split(",")
                    self.Predictions.append([int(lbl), int(n_sig), int(n_bkg), float(sig)])

        with open("/cr/data01/filip/ROC_curves/" + file_path, "r") as input:
            data_by_line = [Prediction(*line.split()) for line in input.readlines()]

        FP_energy, TP_energy, FN_energy, TN_energy = [], [], [], []
        FP_zenith, TP_zenith, FN_zenith, TN_zenith = [], [], [], []
        FP_n_sig, TP_n_sig, FN_n_sig, TN_n_sig = [], [], [], []
        FP_n_bkg, TP_n_bkg, FN_n_bkg, TN_n_bkg = [], [], [], []
        FP_SPD, TP_SPD, FN_SPD, TN_SPD = [], [], [], []
        FP_sig, TP_sig, FN_sig, TN_sig = [], [], [], []

        for Station in data_by_line:

            for prediction in Station.Predictions:
                lbl, n_sig, n_bkg, sig = prediction

                if sig <= 0:
                    continue
                
                if lbl == 0:

                    # False negatives
                    if n_sig != 0:
                        FN_energy.append(Station.Energy)
                        FN_SPD.append(Station.SPD)
                        FN_zenith.append(Station.Zenith)
                        FN_n_sig.append(n_sig)
                        FN_n_bkg.append(n_bkg)
                        FN_sig.append(sig)

                    # True negatives
                    else:
                        TN_energy.append(Station.Energy)
                        TN_SPD.append(Station.SPD)
                        TN_zenith.append(Station.Zenith)
                        TN_n_sig.append(n_sig)
                        TN_n_bkg.append(n_bkg)
                        TN_sig.append(sig)
                else:
                    
                    # True positives
                    if n_sig != 0:
                        TP_energy.append(Station.Energy)
                        TP_SPD.append(Station.SPD)
                        TP_zenith.append(Station.Zenith)
                        TP_n_sig.append(n_sig)
                        TP_n_bkg.append(n_bkg)
                        TP_sig.append(sig)

                    # False positives
                    else:
                        FP_energy.append(Station.Energy)
                        FP_SPD.append(Station.SPD)
                        FP_zenith.append(Station.Zenith)
                        FP_n_sig.append(n_sig)
                        FP_n_bkg.append(n_bkg)
                        FP_sig.append(sig)

        # signal strength accuracy
        if task == "signal_efficiency":
            hits, misses = TP_sig, FN_sig
            min_bin = min( FP_sig + TP_sig + FN_sig + TN_sig )
            max_bin = max( FP_sig + TP_sig + FN_sig + TN_sig )
            
            bins = np.geomspace(min_bin, max_bin, 60)

            hits_histogrammed, _ = np.histogram(hits, bins = bins)
            misses_histogrammed, sig = np.histogram(misses, bins = bins)

            for i, (x, o) in enumerate(zip(hits_histogrammed, misses_histogrammed)):

                if x != 0:
                    accuracy = x / (x+o)
                    accuracy_err = 1/(x+o)**2 * np.sqrt(x * o**2 + o * x**2 + 2 * x*o)
                else:
                    accuracy = accuracy_err = 0

                plt.errorbar((sig[:-1] + sig[:1])[i], accuracy, yerr = accuracy_err, capsize = 5, marker = "o", c = kwargs.get("color", "steelblue"))

            plt.scatter([],[], label = kwargs.get("label","Classifier performance"), c = kwargs.get("color", "steelblue"))
            plt.xscale("log")

        # accuracy over shower plane distance
        if task == "SPD_accuracy":
            hits, misses = TP_SPD + TN_SPD, FP_SPD + FN_SPD
            min_bin = min( FP_SPD + TP_SPD + FN_SPD + TN_SPD )
            max_bin = max( FP_SPD + TP_SPD + FN_SPD + TN_SPD )
            
            bins = np.linspace(min_bin, max_bin, 60)

            hits_histogrammed, _ = np.histogram(hits, bins = bins)
            misses_histogrammed, sig = np.histogram(misses, bins = bins)

            for i, (x, o) in enumerate(zip(hits_histogrammed, misses_histogrammed)):

                if x == 0:
                    continue
                else:

                    accuracy = x / (x+o)
                    accuracy_err = 1/(x+o)**2 * np.sqrt(x * o**2 + o * x**2 + 2 * x*o)

                    plt.errorbar((sig[:-1] + sig[:1])[i], accuracy, yerr = accuracy_err, capsize = 5, marker = "o", c = kwargs.get("color", "steelblue"))

            plt.scatter([],[], label = kwargs.get("label","Classifier performance"), c = kwargs.get("color", "steelblue"))
            plt.xlabel("Shower plane distance / m")
            plt.ylabel("Accuracy")

        # sensitivity over shower plane distance
        if task == "SPD_sensitivity":
            hits, misses = TP_SPD, FN_SPD
            min_bin = min( FP_SPD + TP_SPD + FN_SPD + TN_SPD )
            max_bin = max( FP_SPD + TP_SPD + FN_SPD + TN_SPD )
            
            bins = np.linspace(min_bin, max_bin, 60)

            hits_histogrammed, _ = np.histogram(hits, bins = bins)
            misses_histogrammed, sig = np.histogram(misses, bins = bins)

            for i, (x, o) in enumerate(zip(hits_histogrammed, misses_histogrammed)):

                if x == 0:
                    continue
                else:

                    accuracy = x / (x+o)
                    accuracy_err = 1/(x+o)**2 * np.sqrt(x * o**2 + o * x**2 + 2 * x*o)

                    plt.errorbar((sig[:-1] + sig[:1])[i], accuracy, yerr = accuracy_err, capsize = 5, marker = "o", c = kwargs.get("color", "steelblue"))

            plt.scatter([],[], label = kwargs.get("label","Classifier performance"), c = kwargs.get("color", "steelblue"))
            plt.xlabel("Shower plane distance / m")
            plt.ylabel("Sensitivity")

        # accuracy over energy
        if task == "energy_accuracy":
            hits, misses = TP_energy + TN_energy, FP_energy + FN_energy
            min_bin = min( FP_energy + TP_energy + FN_energy + TN_energy )
            max_bin = max( FP_energy + TP_energy + FN_energy + TN_energy )
            
            bins = np.geomspace(min_bin, max_bin, 60)

            hits_histogrammed, _ = np.histogram(hits, bins = bins)
            misses_histogrammed, sig = np.histogram(misses, bins = bins)

            for i, (x, o) in enumerate(zip(hits_histogrammed, misses_histogrammed)):

                if x == 0:
                    continue
                else:

                    accuracy = x / (x+o)
                    accuracy_err = 1/(x+o)**2 * np.sqrt(x * o**2 + o * x**2 + 2 * x*o)

                    plt.errorbar((sig[:-1] + sig[:1])[i], accuracy, yerr = accuracy_err, capsize = 5, marker = "o", c = kwargs.get("color", "steelblue"))

            plt.scatter([],[], label = kwargs.get("label","Classifier performance"), c = kwargs.get("color", "steelblue"))
            plt.xlabel("Shower energy / eV")
            plt.ylabel("Accuracy")
            plt.xscale("log")

        # signal bin / background bin heatmap
        if task == "bin_comparison":

            hits_sig = TP_n_sig + TN_n_sig
            hits_bkg = TP_n_bkg + TN_n_bkg
            misses_sig = FP_n_sig + FN_n_sig
            misses_bkg = FP_n_bkg + FN_n_bkg

            # apparently theres no signal bigger than 66 bins
            n_max_backgrounds = 66

            xx, yy = np.meshgrid(range(121), range(n_max_backgrounds + 1))
            data = np.zeros_like(xx)

            for sig, bkg in zip(hits_sig, hits_bkg):
                if bkg > n_max_backgrounds:
                    continue
                else : data[bkg][sig] += 1

            for sig, bkg in zip(misses_sig, misses_bkg):
                if bkg > n_max_backgrounds:
                    continue
                else : data[bkg][sig] -= 1

            data = data.astype("float")
            data[data == 0] = np.nan

            # np.savetxt("/cr/users/filip/Trigger/data.txt", data)

            plt.figure()
            plt.title(kwargs.get("label",""))
            img = plt.imshow(np.arctan(data), extent = (0, 120, 0, 68), cmap = "coolwarm", interpolation = "antialiased", origin = "lower")
            cbar = plt.colorbar(img)
            cbar.set_label(r"surplus misses $\leftrightarrow$ surplus hits", labelpad=20)
            cbar.set_ticks([])

            plt.xlabel("# signal bins")
            plt.ylabel("# background bins")