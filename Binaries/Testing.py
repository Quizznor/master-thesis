from .__config__ import *
from .Signal import *
from .Generator import *
from .Classifier import *

# let a classifier predict labels of a given dataset, save predictions in save_dir
def make_dataset(Classifier : Classifier, Dataset : Generator, save_dir : str) -> float :

    from .Classifier import Ensemble
    
    Dataset.__reset__()                 # make sure we're not iterating over an empty set

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

# plot the estimated confidence range of provided classifiers
def confidence_comparison(confidence_level, *args, **kwargs):

    y_max = kwargs.get("ymax", 2500)
    labels = kwargs.get("labels", None)
    energy_labels = ["16_16.5", "16.5_17", "17_17.5", "17.5_18", "18_18.5", "18.5_19", "19_19.5"]
    theta_labels = [r"$0^\circ$", r"$26^\circ$", r"$38^\circ$", r"$49^\circ$", r"$60^\circ$", r"$90^\circ$"]
    colors = ["steelblue", "orange", "green", "maroon", "lime", "indigo", "slategray"]

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

    plt.ylim(-100, y_max)