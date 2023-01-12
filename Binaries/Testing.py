from .__config__ import *
from .Signal import *
from .Generator import *
from .Classifier import *

def make_dataset(Classifier : Classifier, Dataset : Generator, save_dir : str) -> float :

    from .Classifier import Ensemble
    
    Dataset.__reset__()

    if not isinstance(Classifier, Ensemble):

        TPs, FPs = 0, 0

        if Classifier.name == "HardwareClassifier": save_path = "/cr/data01/filip/models/HardwareClassifier/ROC_curve/" + save_dir
        else: save_path = "/cr/data01/filip/models/" + Classifier.name + f"/model_{Classifier.epochs}/ROC_curve/" + save_dir

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
        with open(save_file["TP"], "w") as TP, \
            open(save_file["TN"], "w") as TN, \
            open(save_file["FP"], "w") as FP, \
            open(save_file["FN"], "w") as FN:

            for batch, (traces, true_labels, metadata) in enumerate(Dataset): 

                try:
                    print(f"Fetching batch {batch + 1}/{Dataset.__len__()}: {100 * (batch/Dataset.__len__()):.2f}% \
                    (TP, FP) = ({TPs}, {FPs}) ~ {TPs/(TPs + FPs) * 100 :.2f}", end = "\r")
                except ZeroDivisionError: print("Starting simulation...", end = "\r")

                for predicted_label, true_label, info in zip(Classifier(traces), true_labels, metadata):

                    Integral, (SignalBins, Energy, SPDistance, Zenith) = info
                    true_label = true_label.argmax()

                    if true_label:
                        if predicted_label:
                            prediction = TP
                            TPs += 1
                        else: prediction = FN
                    
                        # save more metadata for traces containing signal
                        save_string = f"{Integral:.3f} {int(SignalBins)} {Energy:.3e} {int(SPDistance)} {Zenith:.3f}"
                    
                    else:
                        if predicted_label:
                            prediction = FP
                            FPs += 1
                        else: prediction = TN

                        # only save deposited charge
                        save_string = f"{Integral:.3f}"

                    prediction.write(save_string + "\n")
        
        
        return TPs / (TPs + FPs)

    else:

        start = perf_counter_ns()

        for i, instance in enumerate(Classifier.models,1):

            time_spent = (perf_counter_ns() - start) * 1e-9
            elapsed = strftime('%H:%M:%S', gmtime(time_spent))
            eta = strftime('%H:%M:%S', gmtime(time_spent * (len(Classifier.models)/i - 1)))

            print(f"Model {i}/{len(Classifier.models)}, {elapsed} elapsed, ETA = {eta}")

            make_dataset(instance, Dataset, save_dir)
            Dataset.__reset__()


def confidence_comparison(confidence_level, *args, **kwargs):

    labels = kwargs.get("labels", None)
    energy_labels = ["16_16.5", "16.5_17", "17_17.5", "17.5_18", "18_18.5", "18.5_19", "19_19.5"]
    theta_labels = [r"$0^\circ$", r"$26^\circ$", r"$38^\circ$", r"$49^\circ$", r"$60^\circ$", r"$90^\circ$"]
    colors = ["steelblue", "orange", "green"]

    try:
        if labels and len(labels) != len(args): raise ValueError
    except:
        sys.exit("Provided labels doesn't match the provided fit parameters")

    fig, axes = plt.subplots(nrows = len(theta_labels) - 1, sharex = True, sharey = True)
    axes[0].set_title(f"Trigger characteristics for r$_{{{confidence_level * 1e2:.0f}}}$")

    for i, fit_info in enumerate(args):
        fit_params, fit_uncertainties = fit_info
        
        for e, energy in enumerate(fit_params):
            for t, theta in enumerate(energy):
                acc, p50, scale = theta
                pcov = fit_uncertainties[e,t]

                station_trigger_probability = lambda x : station_hit_probability(x, acc, p50, scale)
                inverse_trigger_probability = lambda y : p50 - np.log(acc/(1-y) - 1) / scale

                # calculate gradient
                exp = lambda x, k, b : np.exp(-k * (x - b))
                d_accuracy = station_trigger_probability(confidence_level) / acc
                d_p50 = acc * scale * exp(confidence_level, scale, p50) / (1 + exp(confidence_level, scale, p50))**2
                d_scale = acc * (p50 - confidence_level) * exp(confidence_level, scale, p50) / (1 + exp(confidence_level, scale, p50))**2
                grad = np.array([d_accuracy, d_p50, d_scale])
                y_err = np.sqrt( grad.T @ pcov @ grad.T )

                axes[t].errorbar(e, inverse_trigger_probability(confidence_level), xerr = 0.5, yerr = y_err, capsize = 3, c = colors[i], elinewidth = 1, fmt = "s")

    axes[0].set_xticks(range(7), energy_labels)

    fig.text(0.5, 0.04, 'Energy range', ha='center', fontsize = 27)
    fig.text(0.04, 0.5, 'Detection radius / m', va='center', rotation='vertical', fontsize = 27)
    
    for i, ax in enumerate(axes):
        if labels: 
            for ii, label in enumerate(labels):
                ax.scatter([], [], marker = "s", c = colors[ii], label = labels[ii])

        ax.legend(title = theta_labels[i] + r"$\leq$ $\theta$ < " + theta_labels[i + 1])
        ax.axhline(0, c = "gray", ls = ":", lw = 2)



'''
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

# spd efficiency w.r.t energy
def spd_energy(save_path : str, **kwargs) -> None :


# # TODO !!!!
# class EnsembleTesting(Ensemble):

#     def __init__(self, Classifier : str) -> None : pass
'''