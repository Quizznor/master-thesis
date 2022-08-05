from TriggerStudyBinaries_v2.__configure__ import *

Estimator = typing.Union[TriggerClassifier, NNClassifier]

def make_dataset(Classifier : Estimator, Dataset : Generator, save_path : str) -> None :

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

            traces, _ = Dataset.__getitem__(batch, full_trace = True)

            print(f"Fetching batch {batch + 1}/{Dataset.__len__()}: {100 * (batch/Dataset.__len__()):.2f}%", end = "...\r")

            for VEMTrace in traces:

                # ignore mock background traces
                if not VEMTrace.has_signal: 
                    continue

                for i in Dataset.__sliding_window__(VEMTrace):

                    pmt_data, n_sig = VEMTrace.get_trace_window((i, i + Dataset.window_length))
                    signal = VEMTrace.integrate(pmt_data)

                    # mislabel low energy traces
                    if Dataset.ignore_low_VEM: n_sig = 0 if signal < Dataset.ignore_low_VEM else n_sig

                    if n_sig:
                        if Classifier(pmt_data):
                            prediction = TP
                        else: prediction = FN

                        # save more metadata for traces containing signal
                        save_string = f"{signal:.3f} {0} {n_sig} {VEMTrace.Energy:.3e} {VEMTrace.SPDistance:.0f} {VEMTrace.Zenith:.3f}"

                    else:
                        if Classifier(pmt_data):
                            prediction = FP
                        else: prediction = TN

                        # only save signal and number of background bins
                        save_string = f"{signal:.3f} {0}"

                    prediction.write(save_string + "\n")


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


# signal strength roc curve
def ROC(save_path : dict, **kwargs) -> None :

    save_file = \
        {
            "TP" : f"/cr/data01/filip/ROC_curves/{save_path}/true_positives.csv",
            "TN" : f"/cr/data01/filip/ROC_curves/{save_path}/true_negatives.csv",
            "FP" : f"/cr/data01/filip/ROC_curves/{save_path}/false_positives.csv",
            "FN" : f"/cr/data01/filip/ROC_curves/{save_path}/false_negatives.csv"
        }

    temp = [[],[],[],[]]

    for i, p in enumerate(["TP", "TN", "FP", "FN"]):
        temp[i] = np.loadtxt(save_file[p], usecols = 0)

    TP, TN, FP, FN = temp
    TPs, TNs, FPs, FNs = len(TP), len(TN), len(FP), len(FN)

    if kwargs.get("full_set", False):
        y, x = np.array(list(TP) + list(TN)), np.array(list(FP) + list(FN))
    else: y, x = TP, FP

    accuracy = ( TNs + TPs ) / ( TPs + TNs + FNs + FPs ) * 100

    print(save_path.ljust(60), f"{f'{TPs}'.ljust(7)} {f'{FPs}'.ljust(7)} {f'{TNs}'.ljust(7)} {f'{FNs}'.ljust(7)} {len(x) + len(y)} -> acc = {accuracy:.2f}%")
    score_low, score_high = min([x.min(), y.min()]) if min([x.min(), y.min()]) > 0 else 0.01, max([x.max(), y.max()])
    last, current_x, current_y = score_low, 0, 0
    ROC_x, ROC_y = [],[]

    for score_bin in np.geomspace(score_low, score_high, kwargs.get("n", 100))[::-1]:

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
    plt.plot(ROC_x, ROC_y, c = kwargs.get("c", "steelblue"), label = " ".join(save_path.split("_")[:2]), ls = kwargs.get("ls", "solid"))
    plt.plot([0,1],[0,1], ls = "--", c = "gray")

# signal precision and recall curve
def PRC(save_path : dict, **kwargs) -> None :

    save_file = \
        {
            "TP" : f"/cr/data01/filip/ROC_curves/{save_path}/true_positives.csv",
            "TN" : f"/cr/data01/filip/ROC_curves/{save_path}/true_negatives.csv",
            "FP" : f"/cr/data01/filip/ROC_curves/{save_path}/false_positives.csv",
            "FN" : f"/cr/data01/filip/ROC_curves/{save_path}/false_negatives.csv"
        }

    temp = [[],[],[]]
    for i, p in enumerate(["TP","FP", "FN"]):

        if os.path.getsize(save_file[p]):
            temp[i] = np.loadtxt(save_file[p], usecols = 0)
        else: temp[i] = np.array([0])

    TP, FP, FN = [np.array(entry) for entry in temp]

    score_low = min([TP.min(), FP.min(), FN.min()]) if min([TP.min(), FP.min(), FN.min()]) > 0 else 0.01
    score_high = max([TP.max(), FP.max(), FN.max()])
    last = score_low
    PRC_x, PRC_y = [],[]

    # len_x, len_y = len(list(TP) + list(FN)), len(list(TP) + list(FP))

    for score_bin in np.geomspace(score_low, score_high, kwargs.get("n", 1000))[::-1]:

        this_tp = ((last > TP) & (TP > score_bin)).sum()
        this_fp = ((last > FP) & (FP > score_bin)).sum()
        this_fn = ((last > FN) & (FN > score_bin)).sum()
        last = score_bin

        if this_tp + this_fn != 0 and this_tp + this_fp != 0:
            this_x = this_tp / (this_tp + this_fn)             # recall
            this_y = this_tp / (this_tp + this_fp)           # precision
        else: continue
        
        PRC_y.append(this_y), PRC_x.append(this_x)

        # print(this_x, this_y)
        # print(current_x, current_y)
    
    # PRC_x, PRC_y = np.array(PRC_x) / PRC_x[-1], - np.array(PRC_y) / PRC_y[-1] + 1
    PRC_x.sort(), PRC_y.sort()

    # plt.xlim(-0.02,1.02)
    # plt.ylim(-0.02,1.02)
    plt.rcParams.update({'font.size': 22})
    plt.xlabel("Recall"), plt.ylabel("Precision")
    plt.plot(PRC_x[1:], PRC_y[1:], c = kwargs.get("c", "steelblue"), label = kwargs.get("label", "Classifier Performance"), ls = kwargs.get("ls", "solid"))
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

    TP_SPD, FN_SPD = [np.loadtxt(save_file[p], usecols = 4) for p in ["TP", "FN"]]
    TP_energy, FN_energy = [np.loadtxt(save_file[p], usecols = 3) for p in ["TP", "FN"]]
    cmap = matplotlib.cm.get_cmap('plasma')

    for j, energy in enumerate(EventGenerator.libraries.keys()):

        c = cmap(j / len(EventGenerator.libraries.keys()))

        low, high = [float(x) for x in energy.split("_")]
        hits_mask = np.where(np.logical_and( TP_energy >= 10 ** low, TP_energy < 10 ** high) )
        misses_mask =  np.where(np.logical_and( FN_energy >= 10 ** low, FN_energy < 10 ** high))

        hits = TP_SPD[hits_mask]
        misses = FN_SPD[misses_mask]

        print(f"{low} < energy < {high} - hits = {len(hits)}, misses = {len(misses)}")

        min_bin = min( [hits.min(), misses.min()] )
        max_bin = max( [hits.max(), misses.max()] )
        
        bins = np.linspace(min_bin, max_bin, 8)

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
    plt.xlabel("Shower plane distance / m")
    plt.ylabel("Efficiency")   

    print()    
