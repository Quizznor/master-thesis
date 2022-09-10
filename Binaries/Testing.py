from .__config__ import *
from .Signal import *
from .Generator import *
from .Classifier import *

import matplotlib.cm as cmap

def make_dataset(Classifier : Classifier, Dataset : Generator, save_dir : str) -> float :

    TPs, FPs = 0, 0
    save_path = "/cr/data01/filip/models/" + Classifier.name + "/ROC_curve/" + save_dir

    if not os.path.isdir(save_path): os.system(f"mkdir -p {save_path}")

    save_file = \
        {
            "TP" : f"{save_path}/true_positives.csv",
            "TN" : f"{save_path}/true_negatives.csv",
            "FP" : f"{save_path}/false_positives.csv",
            "FN" : f"{save_path}/false_negatives.csv"
        }

    # overwrite old data
    for file in save_file.values():
        if os.path.isfile(file): os.system(f"rm -rf {file}")

        os.system(f"touch {file}")

    # open all files only once, increases performance
    with open(save_file["TP"], "a") as TP, \
         open(save_file["TN"], "a") as TN, \
         open(save_file["FP"], "a") as FP, \
         open(save_file["FN"], "a") as FN:

        for batch in range(Dataset.__len__()):

            traces, _ = Dataset.__getitem__(batch, full_trace = True)

            print(f"Fetching batch {batch + 1}/{Dataset.__len__()}: {100 * (batch/Dataset.__len__()):.2f}%", end = "...\r")

            for VEMTrace in traces:

                # ignore mock background traces
                if not VEMTrace.has_signal: 
                    continue

                for i in Dataset.__sliding_window__(VEMTrace):

                    pmt_data, n_sig, integral = VEMTrace.get_trace_window((i, i + Dataset.window_length))

                    # mislabel low energy traces
                    if Dataset.ignore_low_VEM: n_sig = 0 if integral < Dataset.ignore_low_VEM else n_sig

                    if n_sig:
                        if Classifier(pmt_data):
                            prediction = TP
                            TPs += 1
                        else: prediction = FN

                        # save more metadata for traces containing signal
                        save_string = f"{integral:.3f} {n_sig} {VEMTrace.Energy:.3e} {VEMTrace.SPDistance:.0f} {VEMTrace.Zenith:.3f}"

                    else:
                        if Classifier(pmt_data):
                            prediction = FP
                            FPs += 1
                        else: prediction = TN

                        # only save signal and number of background bins
                        save_string = f"{integral:.3f}"

                    prediction.write(save_string + "\n")
    
    return TPs / (TPs + FPs)


# trigger efficiency over signal
def signal_efficiency(save_path : str, **kwargs) -> None : 

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
def spd_efficiency(save_path : str, **kwargs) -> None : 

    save_file = \
        {
            "TP" : f"/cr/data01/filip/ROC_curves/{save_path}/true_positives.csv",
            "TN" : f"/cr/data01/filip/ROC_curves/{save_path}/true_negatives.csv",
            "FP" : f"/cr/data01/filip/ROC_curves/{save_path}/false_positives.csv",
            "FN" : f"/cr/data01/filip/ROC_curves/{save_path}/false_negatives.csv"
        }

    TP_SPD, FN_SPD = [np.loadtxt(save_file[p], usecols = 3) for p in ["TP", "FN"]]

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
    plt.ylabel("Efficiency")


# accuracy over energy
def energy_accuracy(save_path : dict, **kwargs) -> None :

    save_file = \
        {
            "TP" : f"/cr/data01/filip/ROC_curves/{save_path}/true_positives.csv",
            "TN" : f"/cr/data01/filip/ROC_curves/{save_path}/true_negatives.csv",
            "FP" : f"/cr/data01/filip/ROC_curves/{save_path}/false_positives.csv",
            "FN" : f"/cr/data01/filip/ROC_curves/{save_path}/false_negatives.csv"
        }

    TP_energy, FN_energy = [np.loadtxt(save_file[p], usecols = 2) for p in ["TP", "FN"]]

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
# TODO rework
def bin_comparison(save_path : str, **kwargs) -> None :

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


# signal strength roc curve (TPR vs FPR)
def ROC(Estimator : Classifier, dataset : dict, **kwargs) -> None :

    save_file = \
        {
            "TP" : f"/cr/data01/filip/models/{Estimator.name}/ROC_curve/{dataset}/true_positives.csv",
            "TN" : f"/cr/data01/filip/models/{Estimator.name}/ROC_curve/{dataset}/true_negatives.csv",
            "FP" : f"/cr/data01/filip/models/{Estimator.name}/ROC_curve/{dataset}/false_positives.csv",
            "FN" : f"/cr/data01/filip/models/{Estimator.name}/ROC_curve/{dataset}/false_negatives.csv"
        }

    temp = [[],[],[],[]]

    if not os.path.isdir(f"/cr/data01/filip/models/{Estimator.name}/ROC_curve/{dataset}"):
        print("Not a valid dataset. Valid datasets are:")
        for file in os.listdir(f"/cr/data01/filip/models/{Estimator.name}/ROC_curve/{dataset}"):
            print("\t", file)
        sys.exit()

    for i, p in enumerate(["TP", "TN", "FP", "FN"]):
        temp[i] = np.loadtxt(save_file[p], usecols = 0)

    TP, TN, FP, FN = temp
    TPs, TNs, FPs, FNs = len(TP), len(TN), len(FP), len(FN)

    if kwargs.get("full_set", False):
        y, x = np.array(list(TP) + list(TN)), np.array(list(FP) + list(FN))
    else: y, x = TP, FP

    x = np.clip(x, a_min = 1e-6, a_max = None)
    y = np.clip(y, a_min = 1e-6, a_max = None)

    TPR = ( TPs ) / ( TPs + FPs ) * 100
    ACC = ( TPs + TNs ) / ( TPs + FPs + TNs + FNs ) * 100

    print(f"{Estimator.name} {dataset}".ljust(70), f"{f'{TPs}'.ljust(7)} {f'{FPs}'.ljust(7)} {f'{TNs}'.ljust(7)} {f'{FNs}'.ljust(7)}{f'{TPs + FPs + TNs + FNs}'.ljust(7)} -> Acc = {ACC:.2f}%, TPR = {TPR:.4f}%")
    score_low, score_high = min([x.min(), y.min()]), max([x.max(), y.max()])
    last, current_x, current_y = score_low, 0, 0
    ROC_x, ROC_y = [],[]

    bins = np.geomspace(score_low, score_high, kwargs.get("n", 100))[::-1]
    norm = len(x) + len(y)

    for score_bin in bins:

        this_x = ((last >= x) & (x > score_bin)).sum()
        this_y = ((last >= y) & (y > score_bin)).sum()
        
        current_x += this_x / norm
        current_y += this_y / norm
        
        ROC_y.append(current_y), ROC_x.append(current_x)

        last = score_bin
    
    ROC_x.append(1), ROC_y.append(1)

    plt.xlim(-0.02,1.02)
    plt.ylim(-0.02,1.02)
    plt.rcParams.update({'font.size': 22})
    plt.xlabel("False positive rate"), plt.ylabel("True positive rate")
    plt.plot(ROC_x, ROC_y, c = kwargs.get("c", "steelblue"), label = Estimator.name + " " + " ".join(dataset.split("_")[:5]), ls = kwargs.get("ls", "solid"), lw = 2)
    # plt.scatter(ROC_x, ROC_y)
    
    plt.plot([0,1],[0,1], ls = "--", c = "gray")

# signal precision and recall curve
def PRC(Estimator : Classifier, dataset : dict, **kwargs) -> None :

    save_file = \
        {
            "TP" : f"/cr/data01/filip/models/{Estimator.name}/ROC_curve/{dataset}/true_positives.csv",
            "TN" : f"/cr/data01/filip/models/{Estimator.name}/ROC_curve/{dataset}/true_negatives.csv",
            "FP" : f"/cr/data01/filip/models/{Estimator.name}/ROC_curve/{dataset}/false_positives.csv",
            "FN" : f"/cr/data01/filip/models/{Estimator.name}/ROC_curve/{dataset}/false_negatives.csv"
        }

    temp = [[],[],[]]
    for i, p in enumerate(["TP","FP", "FN"]):

        if os.path.getsize(save_file[p]):
            temp[i] = np.loadtxt(save_file[p], usecols = 0)
        else: temp[i] = np.array([0])

    TP, FP, FN = [np.array(entry) for entry in temp]

    TP = np.clip(TP, a_min = 1e-6, a_max = None)
    FP = np.clip(FP, a_min = 1e-6, a_max = None)
    FN = np.clip(FN, a_min = 1e-6, a_max = None)

    score_low = min([TP.min(), FP.min(), FN.min()])
    score_high = max([TP.max(), FP.max(), FN.max()])
    last = score_low
    PRC_x, PRC_y = [],[]

    for score_bin in np.geomspace(score_low, score_high, kwargs.get("n", 100))[::-1]:

        this_tp = ((last >= TP) & (TP > score_bin)).sum()
        this_fp = ((last >= FP) & (FP > score_bin)).sum()
        this_fn = ((last >= FN) & (FN > score_bin)).sum()
        last = score_bin

        if this_tp + this_fn != 0 and this_tp + this_fp != 0:
            this_x = this_tp / (this_tp + this_fn)                                  # recall
            this_y = this_tp / (this_tp + this_fp)                                  # precision

        else: continue

        # print(f"{last:.2e} -> {score_bin:.2e}: {this_x}, {this_y}")
        
        PRC_y.append(this_y), PRC_x.append(this_x)

    PRC_y.sort()
    PRC_x.sort()

    plt.xlim(-0.02,1.02)
    plt.ylim(0.48,1.02)
    plt.rcParams.update({'font.size': 22})
    plt.xlabel("Efficiency"), plt.ylabel("Precision")
    plt.plot(1 - np.array(PRC_x), PRC_y, c = kwargs.get("c", "steelblue"), label = Estimator.name + " " + " ".join(dataset.split("_")[:5]), ls = kwargs.get("ls", "solid"))
    plt.plot([0,1],[0.5,0.5,], ls = "--", c = "gray")

# spd efficiency w.r.t energy
def spd_energy(save_path : str, **kwargs) -> None :

    print(save_path, ":")

    ls = kwargs.get("ls","solid")
    label = kwargs.get("label", "data")
    marker = kwargs.get("marker", "o")

    save_file = \
        {
            "TP" : f"/cr/data01/filip/ROC_curves/{save_path}/true_positives.csv",
            "TN" : f"/cr/data01/filip/ROC_curves/{save_path}/true_negatives.csv",
            "FP" : f"/cr/data01/filip/ROC_curves/{save_path}/false_positives.csv",
            "FN" : f"/cr/data01/filip/ROC_curves/{save_path}/false_negatives.csv"
        }

    TP_SPD, FN_SPD = [np.loadtxt(save_file[p], usecols = 3) for p in ["TP", "FN"]]
    TP_energy, FN_energy = [np.loadtxt(save_file[p], usecols = 2) for p in ["TP", "FN"]]
    map = cmap.get_cmap('plasma')

    for j, energy in enumerate(EventGenerator.libraries.keys()):

        c = map(j / len(EventGenerator.libraries.keys()))

        low, high = [float(x) for x in energy.split("_")]
        hits_mask = np.where(np.logical_and( TP_energy >= 10 ** low, TP_energy < 10 ** high) )
        misses_mask =  np.where(np.logical_and( FN_energy >= 10 ** low, FN_energy < 10 ** high))

        hits = TP_SPD[hits_mask]
        misses = FN_SPD[misses_mask]

        print(f"{low} < energy < {high} - hits = {len(hits)}, misses = {len(misses)}")

        min_bin = min( [hits.min(), misses.min()] )
        max_bin = 2000
        
        bins = np.linspace(min_bin, max_bin, 18)

        hits_histogrammed, _ = np.histogram(hits, bins = bins)
        misses_histogrammed, sig = np.histogram(misses, bins = bins)

        x_vals, y_vals, yerr = [], [], []

        for i, (x, o) in enumerate(zip(hits_histogrammed, misses_histogrammed)):

            if x == 0 and o == 0:
                continue
            else:
                x_vals.append(((sig[1:] + sig[:-1])/2)[i])
                y_vals.append(x / (x+o))
                yerr.append( 1/(x+o)**2 * np.sqrt(x * o**2 + o * x**2 + 2 * x*o) )

        plt.errorbar(x_vals, y_vals, yerr = yerr, color = c, lw = 2, capsize = 5, ls = ls, marker = marker)

    plt.plot([],[], ls = ls, label = label, marker = marker, c = "k")
    plt.ylim(0, 1.05)
    plt.xlim(0, 2100)
    plt.xlabel("Shower plane distance / m")
    plt.ylabel("Efficiency")   

    print()    

# # TODO !!!!
# class EnsembleTesting(Ensemble):

#     def __init__(self, Classifier : str) -> None : pass
