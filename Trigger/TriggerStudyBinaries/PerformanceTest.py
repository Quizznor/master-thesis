import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import typing
import os

# custom modules for specific usecase
from TriggerStudyBinaries.Classifier import TriggerClassifier, NNClassifier
from TriggerStudyBinaries.Signal import VEMTrace, Background, Baseline
from TriggerStudyBinaries.Generator import EventGenerator, Generator
Estimator = typing.Union[TriggerClassifier, NNClassifier]

def make_dataset(Classifier : Estimator, Dataset : Generator, save_path : str) -> typing.NoReturn :

    os.system(f"mkdir /cr/data01/filip/ROC_curves/{save_path}")

    save_file = \
        {
            "TP" : f"/cr/data01/filip/ROC_curves/{save_path}/true_positives.csv",
            "TN" : f"/cr/data01/filip/ROC_curves/{save_path}/true_negatives.csv",
            "FP" : f"/cr/data01/filip/ROC_curves/{save_path}/false_positives.csv",
            "FN" : f"/cr/data01/filip/ROC_curves/{save_path}/false_negatives.csv"
        }
    
    for file in save_file.values():
        os.system(f"touch {file}")

    # open all files only once, increases performance
    with open(save_file["TP"], "a") as TP, \
         open(save_file["TN"], "a") as TN, \
         open(save_file["FP"], "a") as FP, \
         open(save_file["FN"], "a") as FN:

        for batch in range(Dataset.__len__()):
            traces, _ = Dataset.__getitem__(batch, reduce = False)

            print(f"Fetching batch {batch + 1}/{Dataset.__len__()}: {100 * (batch/Dataset.__len__()):.2f}%", end = "...\r")

            for Trace in traces:

                # ignore mock background traces
                if Trace.SPDistance == -1:
                    continue

                for i in range(0, Trace.trace_length - Dataset.window_length, Dataset.window_step):

                    n_sig, n_bkg = Trace.get_n_signal_background_bins(i, Dataset.window_length)
                    true_label, pmt_data = Trace.get_trace_window(i, Dataset.window_length, Dataset.ignore_low_VEM)
                    signal = Trace.integrate(pmt_data)

                    if true_label:
                        if Classifier(pmt_data):
                            prediction = TP
                        else: prediction = FN

                        # save more metadata for traces containing signal
                        save_string = f"{signal:.3f} {n_bkg} {n_sig} {Trace.Energy:.3e} {Trace.SPDistance:.0f} {Trace.Zenith:.3f}"

                    else:
                        if Classifier(pmt_data):
                            prediction = FP
                        else: prediction = TN

                        # only save signal and number of background bins
                        save_string = f"{signal:.3f} {n_bkg}"

                    prediction.write(save_string + "\n")


# trigger efficiency over signal
def signal_efficiency(save_path : str, **kwargs) -> typing.NoReturn : 

    save_file = \
        {
            "TP" : f"/cr/data01/filip/ROC_curves/{save_path}/true_positives.csv",
            "TN" : f"/cr/data01/filip/ROC_curves/{save_path}/true_negatives.csv",
            "FP" : f"/cr/data01/filip/ROC_curves/{save_path}/false_positives.csv",
            "FN" : f"/cr/data01/filip/ROC_curves/{save_path}/false_negatives.csv"
        }

    TP_sig, FN_sig = [np.loadtxt(save_file[p], usecols = 0) for p in ["TP","FN"]]

    hits, misses = TP_sig, FN_sig
    min_bin = min( TP_sig + FN_sig )
    max_bin = max( TP_sig + FN_sig )
    
    bins = np.geomspace(min_bin, max_bin, 60)

    hits_histogrammed, _ = np.histogram(hits, bins = bins)
    misses_histogrammed, sig = np.histogram(misses, bins = bins)

    for i, (x, o) in enumerate(zip(hits_histogrammed, misses_histogrammed)):

        if x == 0 and o == 0:
            continue
        else:
            accuracy = x / (x+o)
            accuracy_err = 1/(x+o)**2 * np.sqrt(x * o**2 + o * x**2 + 2 * x*o)     

        plt.errorbar((sig[:-1] + sig[:1])[i], accuracy, yerr = accuracy_err, capsize = 5, marker = "o", c = kwargs.get("c", "steelblue"), ls = kwargs.get("ls", "solid"))

    plt.rcParams.update({'font.size': 22})
    plt.scatter([],[], label = kwargs.get("label","Classifier performance"), c = kwargs.get("c", "steelblue"), ls = kwargs.get("ls", "solid"))
    plt.xscale("log")


# sensitivity over shower plane distance
def spd_efficiency(save_path : str, **kwargs) -> typing.NoReturn : 

    save_file = \
        {
            "TP" : f"/cr/data01/filip/ROC_curves/{save_path}/true_positives.csv",
            "TN" : f"/cr/data01/filip/ROC_curves/{save_path}/true_negatives.csv",
            "FP" : f"/cr/data01/filip/ROC_curves/{save_path}/false_positives.csv",
            "FN" : f"/cr/data01/filip/ROC_curves/{save_path}/false_negatives.csv"
        }

    TP_SPD, FN_SPD = [np.loadtxt(save_file[p], usecols = 4) for p in ["TP", "FN"]]

    hits, misses = TP_SPD, FN_SPD
    min_bin = min( TP_SPD + FN_SPD )
    max_bin = max( TP_SPD + FN_SPD )
    
    bins = np.linspace(min_bin, max_bin, 60)

    hits_histogrammed, _ = np.histogram(hits, bins = bins)
    misses_histogrammed, sig = np.histogram(misses, bins = bins)

    for i, (x, o) in enumerate(zip(hits_histogrammed, misses_histogrammed)):

        if x == 0 and o == 0:
            continue
        else:
            accuracy = x / (x+o)
            accuracy_err = 1/(x+o)**2 * np.sqrt(x * o**2 + o * x**2 + 2 * x*o)     
    
        plt.errorbar((sig[:-1] + sig[:1])[i], accuracy, yerr = accuracy_err, capsize = 5, marker = "o", c = kwargs.get("c", "steelblue"))

    plt.rcParams.update({'font.size': 22})
    plt.scatter([],[], label = kwargs.get("label","Classifier performance"), c = kwargs.get("c", "steelblue"))
    plt.xlabel("Shower plane distance / m")
    plt.ylabel("Sensitivity")


# accuracy over energy
def energy_accuracy(save_path : dict, **kwargs) -> typing.NoReturn :

    save_file = \
        {
            "TP" : f"/cr/data01/filip/ROC_curves/{save_path}/true_positives.csv",
            "TN" : f"/cr/data01/filip/ROC_curves/{save_path}/true_negatives.csv",
            "FP" : f"/cr/data01/filip/ROC_curves/{save_path}/false_positives.csv",
            "FN" : f"/cr/data01/filip/ROC_curves/{save_path}/false_negatives.csv"
        }

    TP_energy, FN_energy = [np.loadtxt(save_file[p], usecols = 3) for p in ["TP", "FN"]]

    hits, misses = TP_energy, FN_energy
    min_bin = min( TP_energy + FN_energy )
    max_bin = max( TP_energy + FN_energy )
    
    bins = np.geomspace(min_bin, max_bin, 60)

    hits_histogrammed, _ = np.histogram(hits, bins = bins)
    misses_histogrammed, sig = np.histogram(misses, bins = bins)

    for i, (x, o) in enumerate(zip(hits_histogrammed, misses_histogrammed)):

        if x == 0 and o == 0:
            continue
        else:
            accuracy = x / (x+o)
            accuracy_err = 1/(x+o)**2 * np.sqrt(x * o**2 + o * x**2 + 2 * x*o)     
            
        plt.errorbar((sig[:-1] + sig[:1])[i], accuracy, yerr = accuracy_err, capsize = 5, marker = "o", c = kwargs.get("c", "steelblue"))

    plt.rcParams.update({'font.size': 22})
    plt.scatter([],[], label = kwargs.get("label","Classifier performance"), c = kwargs.get("c", "steelblue"))
    plt.xlabel("Shower energy / eV")
    plt.ylabel("Accuracy")
    plt.xscale("log")


# signal bin / background bin heatmap
def bin_comparison(save_path : str, **kwargs) -> typing.NoReturn :

    save_file = \
        {
            "TP" : f"/cr/data01/filip/ROC_curves/{save_path}/true_positives.csv",
            "TN" : f"/cr/data01/filip/ROC_curves/{save_path}/true_negatives.csv",
            "FP" : f"/cr/data01/filip/ROC_curves/{save_path}/false_positives.csv",
            "FN" : f"/cr/data01/filip/ROC_curves/{save_path}/false_negatives.csv"
        }

    TP_n_sig, FN_n_sig = [np.loadtxt(save_file[p], usecols = 2) for p in ["TP", "FN"]]
    TP_n_bkg, FN_n_bkg = [np.loadtxt(save_file[p], usecols = 1) for p in ["TP", "FN"]]

    hits_sig = TP_n_sig
    hits_bkg = TP_n_bkg
    misses_sig = FN_n_sig
    misses_bkg = FN_n_bkg

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

    plt.rcParams.update({'font.size': 22})
    plt.figure()
    plt.title(kwargs.get("label",""))
    img = plt.imshow(np.arctan(data), extent = (0, 120, 0, 68), cmap = "coolwarm", interpolation = "antialiased", origin = "lower")
    cbar = plt.colorbar(img)
    cbar.set_label(r"surplus misses $\leftrightarrow$ surplus hits", labelpad=20)
    cbar.set_ticks([])

    plt.xlabel("# signal bins")
    plt.ylabel("# background bins")


# signal strength roc curve
def signal_ROC(save_path : dict, **kwargs) -> typing.NoReturn :

    save_file = \
        {
            "TP" : f"/cr/data01/filip/ROC_curves/{save_path}/true_positives.csv",
            "TN" : f"/cr/data01/filip/ROC_curves/{save_path}/true_negatives.csv",
            "FP" : f"/cr/data01/filip/ROC_curves/{save_path}/false_positives.csv",
            "FN" : f"/cr/data01/filip/ROC_curves/{save_path}/false_negatives.csv"
        }

    x, y = [np.loadtxt(save_file[p], usecols = 0) for p in ["TP", "FP"]]

    score_low, score_high = min([x.min(), y.min()]) if min([x.min(), y.min()]) > 0 else 0.01, max([x.max(), y.max()])
    last, current_x, current_y = score_low, 0, 0
    ROC_x, ROC_y = [],[]

    for score_bin in np.geomspace(score_low, score_high, kwargs.get("n", 1000))[::-1]:

        this_x = ((last > x) & (x > score_bin)).sum()
        this_y = ((last > y) & (y > score_bin)).sum()
        
        ROC_y.append(current_y), ROC_x.append(current_x)
        current_x += this_x / len(x)
        current_y += this_y / len(y)

        last = score_bin
    
    ROC_x.append(1), ROC_y.append(1)

    plt.xlim(-0.02,1.02)
    plt.ylim(-0.02,1.02)
    plt.rcParams.update({'font.size': 22})
    plt.xlabel("False positive rate"), plt.ylabel("True positive rate")
    plt.plot(ROC_x, ROC_y, c = kwargs.get("c", "steelblue"), label = kwargs.get("label", "Classifier Performance"), ls = kwargs.get("ls", "solid"))
    plt.plot([0,1],[0,1], ls = "--", c = "gray")