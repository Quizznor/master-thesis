from abc import abstractmethod

from .__config__ import *
from .Signal import *
from .Generator import *

# Some general definitiones that all classifiers should have
class Classifier():

    @abstractmethod
    def __init__(self, name : str) -> None : self.name = name

    @abstractmethod
    def __call__(self) -> int : raise NotImplementedError

    # test the trigger rate of the classifier on random traces
    def production_test(self, n_traces : int = 10000, **kwargs) -> tuple :

        n_total_triggered = 0
        start_time = str(datetime.now()).replace(" ", "-")[:-10]

        os.system(f"mkdir -p /cr/users/filip/plots/production_tests/{self.name.replace('/','-')}/{start_time}/")

        window_length = 120
        window_step = int(window_length / 3)

        RandomTraces = EventGenerator("19_19.5", split = 1, force_inject = 0, real_background = True, prior = 0, window_length = window_length, window_step = window_step, **kwargs)
        RandomTraces.files = np.zeros(n_traces)
        total_trace_duration = GLOBAL.trace_length * GLOBAL.single_bin_duration * len(RandomTraces)

        start = perf_counter_ns()

        for batch, trace in enumerate(RandomTraces):

            progress_bar(batch, n_traces, start)

            for window in trace[0]:

                if self.__call__(window):

                    n_total_triggered += 1

                    if n_total_triggered < 100: 
                        self.plot_trace_window(window, n_total_triggered, start_time, trace[0].downsampled)

                    # perhaps skipping the entire trace isn't exactly accurate
                    # but then again just skipping one window seems wrong also
                    # -> David thinks this should be fine
                    break

        trigger_frequency = n_total_triggered / total_trace_duration
        frequency_error = np.sqrt(n_total_triggered) / total_trace_duration

        print("\n\nProduction test results:")
        print("")
        print(f"random traces injected: {n_traces}")
        print(f"summed traces duration: {total_trace_duration:.4}s")
        print(f"total T2 trigger found: {n_total_triggered}")
        print(f"*********************************")
        print(f"TRIGGER FREQUENCY = {trigger_frequency:.2f} +- {frequency_error:.2f} Hz")

        return trigger_frequency, frequency_error, n_traces, n_total_triggered, total_trace_duration

    # helper function that saves triggered trace windows for production_test()
    def plot_trace_window(self, trace : np.ndarray, index : int, start_time : str, downsampled : bool) -> None : 

        (pmt1, pmt2, pmt3), x = trace, len(trace[0])
        assert len(pmt1) == len(pmt2) == len(pmt3), "TRACE LENGTHS DO NOT MATCH"

        plt.rcParams["figure.figsize"] = [20, 10]

        plt.plot(range(x), pmt1, label = r"PMT \#1")
        plt.plot(range(x), pmt2, label = r"PMT \#2")
        plt.plot(range(x), pmt3, label = r"PMT \#3")

        

        plt.ylabel(r"Signal / VEM$_\mathrm{Peak}$")
        plt.xlabel(r"Bin / $8.33\,\mathrm{ns}$" if not downsampled else r"Bin / $25\,\mathrm{ns}$")
        plt.xlim(0, x)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(f"/cr/users/filip/plots/production_tests/{self.name.replace('/','-')}/{start_time}/trigger_{index}")
        plt.cla()

    # calculate performance for a given set of showers
    def make_signal_dataset(self, Dataset : "Generator", save_dir : str, n_showers : int = None, save_traces : bool = False) -> None : 

        if save_traces: raise NotImplementedError("Not implemented at the moment due to rework")

        temp, Dataset.for_training = Dataset.for_training, False
        if n_showers is None: n_showers = Dataset.__len__()

        if self.name == "HardwareClassifier": save_path = "/cr/data01/filip/models/HardwareClassifier/ROC_curve/" + save_dir
        else: save_path = "/cr/data01/filip/models/" + self.name + f"/model_{self.epochs}/ROC_curve/" + save_dir

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        start_time = perf_counter_ns()
        save_file = \
            {
                "TP" : f"{save_path}/true_positives.csv",
                "TN" : f"{save_path}/true_negatives.csv",
                "FP" : f"{save_path}/false_positives.csv",
                "FN" : f"{save_path}/false_negatives.csv"
            }
        
        # open all files only once, increases performance
        with open(save_file["TP"], "w") as TP, \
             open(save_file["TN"], "w") as TN, \
             open(save_file["FP"], "w") as FP, \
             open(save_file["FN"], "w") as FN:

            for batch_no, traces in enumerate(Dataset):

                progress_bar(batch_no, n_showers, start_time)
                random_file = Dataset.files[batch_no]
                random_file = f"{'/'.join(random_file.split('/')[-3:])}"

                for trace in traces:

                    StationID = trace.StationID
                    SPDistance = trace.SPDistance
                    Energy = trace.Energy
                    Zenith = trace.Zenith
                    n_muons = trace.n_muons
                    n_electrons = trace.n_electrons
                    n_photons = trace.n_photons

                    save_string = f"{random_file} {StationID} {SPDistance} {Energy} {Zenith} {n_muons} {n_electrons} {n_photons} "
                    max_charge_integral = -np.inf

                    for window in trace:

                        # we inherently care more about the situation were we trigger as
                        # these events are seen by CDAS. Thus handle FP/TP more strictly

                        label, integral = Dataset.calculate_label(trace)
                        has_triggered = self.__call__(window)

                        # we have detected a false positive in the trace
                        if label != "SIG" and has_triggered:
                            FP.write(save_string + f"{integral}\n")
                        
                        # we have detected a true positive, break loop
                        elif label == "SIG" and has_triggered:
                            TP.write(save_string + f"{integral}\n")
                            break

                        # we haven't seen anything, keep searching
                        else:
                            if integral > max_charge_integral:
                                max_charge_integral = integral

                            # TODO: TN handling would go here

                    # loop didn't break, we didn't see shit
                    else:
                        FN.write(save_string + f"{max_charge_integral}\n")

                if batch_no + 1 >= n_showers: break        

        Dataset.for_training = temp
        Dataset.__reset__()

    # combine multiple results from self.make_signal_dataset to one
    def combine_data(self, save_dir : str, *args) -> None : 

        if self.name == "HardwareClassifier":  root_path = "/cr/data01/filip/models/HardwareClassifier/ROC_curve/"
        else: root_path = "/cr/data01/filip/models/" + self.name + f"/model_{self.epochs}/ROC_curve/"

        os.mkdir(root_path + save_dir)
        warnings.simplefilter("ignore", UserWarning)
        
        for prediction in ["true_positives.csv", "true_negatives.csv", "false_positives.csv", "false_negatives.csv"]:

            os.system(f"touch {root_path}/{save_dir}/{prediction}")
            with open(root_path + save_dir + "/" + prediction, "a") as target:
                for dataset in args:
                    with open(root_path + dataset + "/" + prediction , "r") as source:
                        for line in source.readlines():
                            target.write(line)

        warnings.simplefilter("default", UserWarning)
    
    # load a specific dataset (e.g. 'validation_data', 'real_background', etc.) and print performance
    def load_and_print_performance(self, dataset : str, quiet = False) -> tuple : 

        if not quiet:
            try:
                if header_was_called: pass

            except NameError:
                self.__header__()

            print(f"Fetching predictions for: {self.name} -> {dataset}", end = "\r")

        # load dataset in it's completeness
        if self.name == "HardwareClassifier":
            save_files = \
            {
                "TP" : f"/cr/data01/filip/models/{self.name}/ROC_curve/{dataset}/true_positives.csv",
                "TN" : f"/cr/data01/filip/models/{self.name}/ROC_curve/{dataset}/true_negatives.csv",
                "FP" : f"/cr/data01/filip/models/{self.name}/ROC_curve/{dataset}/false_positives.csv",
                "FN" : f"/cr/data01/filip/models/{self.name}/ROC_curve/{dataset}/false_negatives.csv"
            }
        else:
            save_files = \
            {
                "TP" : f"/cr/data01/filip/models/{self.name}/model_{self.epochs}/ROC_curve/{dataset}/true_positives.csv",
                "TN" : f"/cr/data01/filip/models/{self.name}/model_{self.epochs}/ROC_curve/{dataset}/true_negatives.csv",
                "FP" : f"/cr/data01/filip/models/{self.name}/model_{self.epochs}/ROC_curve/{dataset}/false_positives.csv",
                "FN" : f"/cr/data01/filip/models/{self.name}/model_{self.epochs}/ROC_curve/{dataset}/false_negatives.csv"
            }

        if os.stat(save_files['TN']).st_size:
            TN = np.loadtxt(save_files['TN'], usecols = [2, 3, 4, 5, 6, 7])
        else: TN = np.array([])

        if os.stat(save_files['FP']).st_size:
            FP = np.loadtxt(save_files['FP'], usecols = [2, 3, 4, 5, 6, 7])
        else: FP = np.array([])

        if os.stat(save_files['TP']).st_size:
            TP = np.loadtxt(save_files['TP'], usecols = [2, 3, 4, 5, 6, 7])
        else: TP = np.array([])
        
        if os.stat(save_files['FN']).st_size:
            FN = np.loadtxt(save_files['FN'], usecols = [2, 3, 4, 5, 6, 7])
        else: FN = np.array([])


        if not quiet:
            tp, fp = len(TP), len(FP)
            tn, fn = len(TN), len(FN)

            ACC = ( tp + tn ) / (tp + fp + tn + fn)* 100

            name = self.name if len(self.name) <= 43 else self.name[:40] + "..."
            dataset = dataset if len(dataset) <= 33 else dataset[:30] + "..."
            print(f"{name:<45} {dataset:<35} {tp:7d} {fp:7d} {tn:7d} {fn:7d} -> {ACC = :6.2f}%")

        return TP, FP, TN, FN

    # Performance visualizers #######################################################################
    if True:                                              # So this can be collapsed in the editor =)
        # plot the classifiers efficiency at a given SPD, energy, and theta
        # TODO error calculation from LDF fit, etc...
        def spd_energy_efficiency(self, dataset : str, **kwargs) -> None :
            
            warnings.simplefilter("ignore", RuntimeWarning)
            colormap = cmap.get_cmap("plasma")
            bar_kwargs = \
            {
                "fmt" : "o",
                "elinewidth" : 0.5,
                "capsize" : 3
            }
            
            e_labels = [r"$16$", r"$16.5$", r"$17$", r"$17.5$", r"$18$", r"$18.5$", r"$19$", r"$19.5$"]            
            annotate = lambda e : e_labels[e] + r" $\leq$ log($E$ / eV) $<$ " + e_labels[e + 1]

            energy_bins = [10**16, 10**16.5, 10**17, 10**17.5, 10**18, 10**18.5, 10**19, 10**19.5]      # uniform in log(E)
            theta_bins =  [0.0000, 33.5600, 44.4200, 51.3200, 56.2500, 65.3700]                         # pseudo-uniform in sec(θ)

            miss_sorted = [[ [] for t in range(len(theta_bins) - 1) ] for e in range(len(energy_bins) - 1)]
            hits_sorted = [[ [] for t in range(len(theta_bins) - 1) ] for e in range(len(energy_bins) - 1)]

            # Prediction structure: []
            TP, FP, TN, FN = self.load_and_print_performance(dataset)

            # Sort predictions into bins of theta and energy
            for source, target in zip([TP, FN], [hits_sorted, miss_sorted]):

                spd, energy, theta = source[:, 0], source[:, 1], source[:, 2]
                energy_sorted = np.digitize(energy, energy_bins)
                theta_sorted = np.digitize(theta, theta_bins)

                for e, t, shower_plane_distance in zip(energy_sorted, theta_sorted, spd):
                    target[e - 1][t - 1].append(shower_plane_distance)


            if kwargs.get("draw_plot", True):
                fig, axes = plt.subplots(3,3, sharex = False, sharey = True, figsize = [50, 25])
                fig.suptitle(f"{self.name} - {dataset}", fontsize = 50)
                axes[-1][-1].axis("off"), axes[-1][-2].axis("off")
                plt.ylim(-0.05, 1.05)

            # Calculate efficiencies given sorted performances
            # axis 1 = sorted by primary particles' energy
            for e, (hits, misses) in enumerate(zip(hits_sorted, miss_sorted)):

                if kwargs.get("draw_plot", True): 
                    ax = axes[e // 3][e % 3]
                    ax.axvline(1500, c = "k", ls = "--")
                    ax.set_xlim(0, 6000),
                    ax.plot([], [], ls = "solid", c = "k", label = "Extrapolated")
                    ax.legend(loc = "upper right", title = annotate(e))
                    if e >= 4: 
                        ax.set_xlabel("Shower plane distance / m")
                        ax.set_xticks([1e3, 2e3, 3e3, 4e3, 5e3])
                    else:
                        ax.set_xticks([])
                    if e % 3 == 0: ax.set_ylabel("Trigger efficiency")

                # axis 2 = sorted by zenith angle
                for t, (hits, misses) in enumerate(zip(hits, misses)):

                    LDF, (LDF_efficiency, LDF_prob_50, LDF_scale) = get_fit_function("/cr/tempdata01/filip/QGSJET-II/LDF/", e, t)

                    c = colormap(t / (len(theta_bins) - 1))
                    all_data = hits + misses
                    n_data_in_bins = int(50 * np.sqrt(e + 1))

                    # have at least 7 bins or bins with >50 samples
                    while True:
                        n_bins = len(all_data) // n_data_in_bins
                        probabilities = np.linspace(0, 1, n_bins)
                        binning = mquantiles(all_data, prob = probabilities)
                        bin_center = 0.5 * (binning[1:] + binning[:-1])
                        n_all, _ = np.histogram(all_data, bins = binning)

                        if len(n_all) <= 7: 
                            n_data_in_bins -= 10
                            if n_data_in_bins == 50: break
                        else: break

                    x, _ = np.histogram(hits, bins = binning)
                    o, _ = np.histogram(misses, bins = binning)
                    efficiency = x / (x + o) * LDF(bin_center)
                    efficiency_err = 1/n_all**2 * np.sqrt( x**3 + o**3 - 2 * np.sqrt((x * o)**3) )          # lack LDF error part here !!
                    efficiency_err[efficiency_err == 0] = 1e-3                                              # such that residuals are finite

                    # perform fit with x_err and y_err
                    try:
                        if kwargs.get("perform_fit", True):

                            try:
                                # # lateral_trigger_probability(x : np.ndarray, efficiency : float, prob_50, scale : float, C : float)
                                # popt, pcov = curve_fit(lateral_trigger_probability, bin_center, efficiency, 
                                #                                         p0 = [efficiency[0], bin_center[np.argmin(abs(efficiency - 50))], LDF_scale, 2 * LDF_scale],
                                #                                         bounds = ([0, 0, 0, 0], [1, np.inf, 1, 0.5]),
                                #                                         sigma = efficiency_err,
                                #                                         absolute_sigma = True,
                                #                                         maxfev = 10000)

                                # When running in compatibility mode (i.e. using lateral_distribution_function under the hood)
                                # lateral_distribution_function(x : np.ndarray, efficiency : float, prob_50 : float, scale : float)
                                popt, pcov = curve_fit(lateral_trigger_probability, bin_center, efficiency, 
                                                                        p0 = [1, bin_center[np.argmin(abs(efficiency - 50))], LDF_scale],
                                                                        bounds = ([0, 0, 0], [np.inf, np.inf, 1]),
                                                                        sigma = efficiency_err,
                                                                        absolute_sigma = True,
                                                                        maxfev = 10000)

                                # write fit parameters to disk
                                file_name = f"{e_labels[e].replace('$','')}_{e_labels[e+1].replace('$','')}__{int(theta_bins[t])}_{int(theta_bins[t+1])}.csv"

                                if hasattr(self, "epochs"):
                                    fit_dir = f"/cr/data01/filip/models/{self.name}/model_{self.epochs}/ROC_curve/{dataset}/FITPARAM/"
                                else: fit_dir = f"/cr/data01/filip/models/{self.name}/ROC_curve/{dataset}/FITPARAM/"
                                
                                try: os.mkdir(fit_dir)
                                except FileExistsError: pass

                                with open(fit_dir + file_name, "w") as fit_parameters:
                                    np.savetxt(fit_parameters, popt)


                                if kwargs.get("draw_plot", True):

                                    X = np.geomspace(1e-3, 6000, 1000)

                                    efficiency_fit = lateral_trigger_probability(X, *popt)
                                    efficiency_fit_error = lateral_trigger_probability_error(X, pcov, *popt)
                                    bottom = np.clip(efficiency_fit - efficiency_fit_error, 0, 1)
                                    top = np.clip(efficiency_fit + efficiency_fit_error, 0, 1)
                                
                                    ax.plot(X, efficiency_fit, color = c)
                                    ax.fill_between(X, bottom, top, color = c, alpha = 0.1)
                            
                            except IndexError: pass
                    
                    except ValueError: pass

                    if kwargs.get("draw_plot", True):
                        upper = np.clip(efficiency + efficiency_err, 0, 1)
                        lower = np.clip(efficiency - efficiency_err, 0, 1)

                        ax.errorbar(bin_center, efficiency, yerr = [efficiency - lower, upper - efficiency], color = c, **bar_kwargs)

            norm = BoundaryNorm(theta_bins, colormap.N)
            ax2 = fig.add_axes([0.92, 0.1, 0.01, 0.8])
            cbar = ColorbarBase(ax2, cmap=colormap, norm=norm, label = r"sec$(\theta)$ - 1")
            cbar.set_ticks(theta_bins)
            cbar.set_ticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.4"])

            plt.subplots_adjust(hspace = 0.04, wspace = 0)

            warnings.simplefilter("default", RuntimeWarning)

        # plot the ROC curve for a specific dataset (e.g. 'validation_data', 'real_background', etc.) over signal strength (VEM_charge)
        def ROC(self, dataset, **kwargs) -> None :

            raise NotImplementedError

            TP, FP, TN, FN = self.load_and_print_performance(dataset, usecols = [0, 0, 0, 0])

            if kwargs.get("full_set", False):
                y, x = np.array(list(TP) + list(TN)), np.array(list(FP) + list(FN))
            else: y, x = TP, FP

            x = np.clip(x, a_min = 1e-6, a_max = None)
            y = np.clip(y, a_min = 1e-6, a_max = None)

            score_low, score_high = min([x.min(), y.min()]), max([x.max(), y.max()])
            last, current_x, current_y = score_low, 0, 0
            ROC_x, ROC_y = [],[]

            bins = np.geomspace(score_low, score_high, kwargs.get("n", 50))[::-1]
            norm = ( len(x) + len(y) ) * 5                  # why x5 ???

            for score_bin in bins:

                this_x = ((last >= x) & (x > score_bin)).sum()
                this_y = ((last >= y) & (y > score_bin)).sum()
                
                current_x += this_x / norm
                current_y += this_y / norm
                
                ROC_y.append(current_y), ROC_x.append(current_x)

                last = score_bin
            
            ROC_x.append(1), ROC_y.append(1)

            plt.title(kwargs.get("title",""))
            plt.xlim(-0.02,1.02)
            plt.ylim(-0.02,1.02)
            plt.rcParams.update({'font.size': 22})
            plt.xlabel("False positive rate"), plt.ylabel("True positive rate")
            plt.plot(ROC_x, ROC_y, label = kwargs.get("label", self.name + "/" + dataset), ls = kwargs.get("ls", "solid"), lw = 2)
            
            plt.plot([0,1],[0,1], ls = "--", c = "gray")

            tp, fp = len(TP), len(FP)
            tn, fn = len(TN), len(FN)

            all = ( tp + tn + fp + fn )
            TPR = ( tp ) / ( tp + fp ) * 100
            ACC = ( tp + tn ) / all * 100

            return ACC, TPR

        # precision and recall curve over signal strength (VEM_charge)
        def PRC(self, dataset : str, **kwargs) -> None :

            raise NotImplementedError

            TP, FP, TN, FN = self.load_and_print_performance(dataset, usecols = [0, 0, 0, 0])

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

            PRC_y.sort() # ???
            PRC_x.sort() # ???

            plt.xlim(-0.02,1.02)
            plt.ylim(0.48,1.02)
            plt.rcParams.update({'font.size': 22})
            plt.xlabel("Efficiency"), plt.ylabel("Precision")
            plt.plot(1 - np.array(PRC_x), PRC_y, c = kwargs.get("c", "steelblue"), label = self.name + "/" + dataset, ls = kwargs.get("ls", "solid"))
            plt.plot([0,1],[0.5,0.5,], ls = "--", c = "gray")

        # plot the classifiers efficiency in terms of deposited signal
        def signal_efficiency(self, dataset : str, **kwargs) -> None : 

            theta_bins = [26, 38, 49, 60]
            energy_bins = [10**16.5, 1e17, 10**17.5, 1e18, 10**18.5, 1e19]
            TP, FP, TN, FN = self.load_and_print_performance(dataset)
            # Prediction structure: [ integral, n_signal, energy, SPD, Theta]

            signal_bins = np.geomspace(1e-1, 1e4, kwargs.get("bins", 50))
            colormap = cmap.get_cmap("plasma")
            
            warnings.simplefilter("ignore", RuntimeWarning)

            miss_sorted = [[ [] for t in range(len(theta_bins) + 1) ] for e in range(len(energy_bins) + 1)]
            hits_sorted = [[ [] for t in range(len(theta_bins) + 1) ] for e in range(len(energy_bins) + 1)]
            e_labels = [r"$16$", r"$16.5$", r"$17$", r"$17.5$", r"$18$", r"$18.5$", r"$19$", r"$19.5$"]
            t_labels = [r"0$^{\circ}$", r"26$^{\circ}$", r"38$^{\circ}$", r"49$^{\circ}$", r"60$^{\circ}$", r"90$^{\circ}$"]

            # sort predictions into bins of theta and energy
            for source, target in zip([TP, FN], [hits_sorted, miss_sorted]):

                SPD, E, T = source[:, 0], source[:, 1], source[:, 2]

                # sort misses / hits w.r.t zenith and primary energy
                theta_indices = np.digitize(T, theta_bins)
                energy_indices = np.digitize(E, energy_bins)

                for e, t, spd in zip(energy_indices, theta_indices, SPD):
                    target[e][t].append(spd)

            for e, (hits_by_energy, misses_by_energy) in enumerate(zip(hits_sorted, miss_sorted)):

                fig = plt.figure()
                plt.xscale("log")
                plt.ylim(-0.05, 1.05)
                plt.errorbar([],[], c = "k", ls = "--", label = "Simulated")
                plt.legend(loc = "upper right", title = e_labels[e] + r" $\leq$ log($E$ / eV) < " + e_labels[e + 1], title_fontsize = 19)
                plt.xlabel("Deposited Signal / VEM")
                plt.ylabel("Trigger efficiency / %")

                for t, (hits_by_theta, miss_by_theta) in enumerate(zip(hits_by_energy, misses_by_energy)):

                    c = colormap(t / len(hits_by_energy))
                    x, _ = np.histogram(hits_by_theta, bins = signal_bins)
                    o, _ = np.histogram(miss_by_theta, bins = signal_bins)

                    bins, y = 0.5 * (signal_bins[1:] + signal_bins[:-1]), x / (x + o)
                    sx, sy = 0.5 * np.diff(signal_bins), x/(x + 0)**2 * np.sqrt(o**2 + (x + 2*o)/(x + o)**2)

                    plt.errorbar(bins, y, ls = "--", xerr = sx, yerr = sy, label = t_labels[t] + r"$\leq$ $\theta$ < " + t_labels[t + 1], color = c)
                    

                norm = BoundaryNorm([0] + theta_bins + [90], colormap.N)
                ax2 = fig.add_axes([0.95, 0.1, 0.01, 0.8])
                ColorbarBase(ax2, cmap=colormap, norm=norm, label = r"Zenith angle")

            plt.show()

        # relate single station efficiency function to T3 efficiency
        def do_t3_simulation(self, dataset : str, n_points : int = 1e5) -> None :

            if isinstance(n_points, float): n_points = int(n_points)

            if isinstance(self, NNClassifier):
                fitparams = np.loadtxt(f"/cr/data01/filip/models/{self.name}/model_{self.epochs}/ROC_curve/{dataset}/fit_params.csv")
            else:
                fitparams = np.loadtxt(f"/cr/data01/filip/models/{self.name}/ROC_curve/{dataset}/fit_params.csv")

            plt.rcParams["figure.figsize"] = [25, 18]
            plt.rcParams["font.size"] = 22
            colormap = cmap.get_cmap("plasma")
            fig, ax = plt.subplots()

            # set up plot
            ax.text( 635, -55, "T3 detected", fontsize = 22)
            ax.text(1395, 775, "T3 missed", fontsize = 22)
            symmetry_line = lambda x : 1500 - x
            X = np.linspace(700, 1550, 100)
            ax.scatter([0, 1500, 750, 2250], [0, 0, 750, 750], s = 100, c = "k")
            ax.plot(X, symmetry_line(X), ls = "solid", c = "k", zorder = 0, lw = 2)
            ax.add_patch(Polygon([[0,0], [1500, 0], [750, 750]], closed = True, color = "green", alpha = 0.1, lw = 0))
            ax.add_patch(Polygon([[750, 750], [1500, 0], [2250, 750]], closed = True, color = "red", alpha = 0.1, lw = 0))

            # create shower cores in target area
            theta_bins = [0, 26, 38, 49, 60, 90]
            ys = np.random.uniform(0, 750, n_points)
            xs = np.random.uniform(0, 1500, n_points) + ys
            reflect = [ ys[i] > symmetry_line(xs[i]) for i in range(len(xs))]
            xs[reflect] = -xs[reflect] + 2250
            ys[reflect] = -ys[reflect] + 750

            start_time = perf_counter_ns()

            # do the T3 simulation
            t3_hits, t3_misses = np.zeros((7, 5)), np.zeros((7, 5))
            x_container, y_container = [[[] for t in range(5)] for e in range(7)], [[[] for t in range(5)] for e in range(7)]
            stations = [[0, 0, 0], [1500, 0, 0], [750, 750, 0]]

            for step_count, (x, y) in enumerate(zip(xs, ys)):

                progress_bar(step_count, n_points, start_time)

                energy_and_theta = np.random.randint(0, len(fitparams))
                energy, t = energy_and_theta // 5, energy_and_theta % 5
                fit_function = lambda spd : lateral_trigger_probability(x, *fitparams[energy_and_theta])

                # choose theta, phi at random, calculate shower_plane_distance
                theta = np.radians(np.random.uniform(theta_bins[t], theta_bins[t + 1]))
                phi = np.random.uniform(0, 2 * np.pi)
                sp_distances = []
                
                for station in stations:

                    core_position = np.array([x, y, 0])
                    core_origin = np.sin(theta) * np.array([np.cos(phi), np.sin(phi), 1/np.tan(theta)]) + core_position

                    shower_axis = core_position - core_origin
                    dot_norm = np.dot(shower_axis, shower_axis)
                    perpendicular_norm = np.dot(station - core_origin, shower_axis) / dot_norm
                    sp_distances.append( np.linalg.norm(perpendicular_norm * shower_axis + (core_origin - station)))

                # #  In case of paranoia regarding distance calculation break comment
                # ax.add_patch(plt.Circle((0, 0), sp_distances[0], color='b', fill=False))
                # ax.add_patch(plt.Circle((1500, 0), sp_distances[1], color='b', fill=False))
                # ax.add_patch(plt.Circle((750, 750), sp_distances[2], color='b', fill=False))

                trigger_probabilities = [fit_function(distance) for distance in sp_distances]
                dice_roll = np.random.uniform(0, 1, 3)

                if np.all(dice_roll < trigger_probabilities):
                    t3_hits[energy][t] += 1
                    # plt.scatter(x, y, c = "k")
                else:
                    x, y = 2250 - x, 750 - y
                    t3_misses[energy][t] += 1
                    # plt.scatter(x, y, c = "r")

                x_container[energy][t].append(x)
                y_container[energy][t].append(y)

            size_bins = [30, 50, 70, 90, 110, 160, 200]
            e_labels = [r"$16$", r"$16.5$", r"$17$", r"$17.5$", r"$18$", r"$18.5$", r"$19$", r"$19.5$"]

            for e, (x_energy, y_energy) in enumerate(zip(x_container, y_container)):
                for t, (x, y) in enumerate(zip(x_energy, y_energy)):

                    c = colormap(t / len(x_energy))
                    s = size_bins[e]

                    ax.scatter(x[::100], y[::100], color = c, s = s, marker = "x")

            for e, bin in enumerate(size_bins):
                ax.scatter([],[], c = "k", s = bin, label = e_labels[e] + r" $\leq$ log($E$ / eV) < " + e_labels[e + 1], marker = "x")

            ax.set_aspect('equal')
            ax.legend(fontsize = 18)
            plt.xlabel("Easting / m")
            plt.ylabel("Northing / m")

            norm = BoundaryNorm(theta_bins, colormap.N)
            ax2 = fig.add_axes([0.91, 0.3, 0.01, 0.4])
            ColorbarBase(ax2, cmap=colormap, norm=norm, label = r"Zenith angle")

            plt.figure()

            e_labels = EventGenerator.libraries.keys()
            t_labels = ["0_26", "26_38", "38_49", "49_60", "60_90"]

            sns.heatmap(t3_hits / (t3_hits + t3_misses) * 1e2, annot = True, fmt = ".1f", cbar_kws = {"label" : "T3 efficiency / %"})
            plt.xticks(ticks = 0.5 + np.arange(0, 5, 1), labels = t_labels)
            plt.yticks(ticks = 0.5 + np.arange(0, 7, 1), labels = e_labels)
            plt.xlabel("Zenith range")
            plt.ylabel("Energy range")

            plt.show()

        @staticmethod
        def __header__() -> None : 

            global header_was_called
            header_was_called = True

            print(f"\n{'Classifier':<45} {'Dataset':<35} {'TP':>7} {'FP':>7} {'TN':>7} {'FN':>7}")

    
    # Performance visualizers #######################################################################

# Wrapper for tf.keras.Sequential model with some additional functionalities
class NNClassifier(Classifier):

    # Early stopping callback that gets evaluated at the end of each batch
    class BatchwiseEarlyStopping(tf.keras.callbacks.Callback):

        def __init__(self, patience : int, acc_threshold : float) -> None :
            self.acc_threshold = acc_threshold
            self.patience = patience

        def __reset__(self) -> None : 
            self.current_patience = 0
            self.best_loss = np.Inf

        def on_batch_end(self, batch, logs : dict = None) -> None :

            if logs.get("accuracy") >= self.acc_threshold:
                current_loss = logs.get("loss")
                if np.less(current_loss, self.best_loss):
                    self.best_loss = current_loss
                    self.current_patience = 0
                else:
                    self.current_patience += 1

                    if self.current_patience >= self.patience: raise EarlyStoppingError

        def on_train_begin(self, logs : dict = None) -> None : self.__reset__()
        def on_epoch_end(self, epoch, logs : dict = None) -> None : 
            self.__reset__()
            print()

    # Collection of different network architectures
    class Architectures():

        ### Library functions to add layers #################
        if True: # so that this can be collapsed in editor =)
            @staticmethod
            def add_input(model, **kwargs) -> None :
                model.add(tf.keras.layers.Input(**kwargs))

            @staticmethod
            def add_dense(model, **kwargs) -> None : 
                model.add(tf.keras.layers.Dense(**kwargs))

            @staticmethod
            def add_conv1d(model, **kwargs) -> None : 
                model.add(tf.keras.layers.Conv1D(**kwargs))

            @staticmethod
            def add_conv2d(model, **kwargs) -> None : 
                model.add(tf.keras.layers.Conv2D(**kwargs))

            @staticmethod
            def add_flatten(model, **kwargs) -> None : 
                model.add(tf.keras.layers.Flatten(**kwargs))

            @staticmethod
            def add_output(model, **kwargs) -> None : 
                model.add(tf.keras.layers.Flatten())
                model.add(tf.keras.layers.Dense(**kwargs))

            @staticmethod
            def add_dropout(model, **kwargs) -> None : 
                model.add(tf.keras.layers.Dropout(**kwargs))

            @staticmethod
            def add_norm(model, **kwargs) -> None : 
                model.add(tf.keras.layers.BatchNormalization(**kwargs))

            @staticmethod
            def add_lstm(**kwargs) -> None :

                # doesn't work, since data is 2-dimensional
                # model.add(tf.keras.layers.LSTM(**kwargs))

                # instead use LSTM for each PMT
                input = tf.keras.layers.Input(kwargs.get("input_shape"))
                unstacked = tf.keras.layers.Lambda(lambda x: tf.unstack(x, axis=1))(input)
                dense_outputs = [tf.keras.layers.LSTM(kwargs.get("d_LSTM", 1), activation = "relu")(x) for x in unstacked]
                merged = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(dense_outputs)
                merged_flatten = tf.keras.layers.Flatten()(merged)

                return tf.keras.Model(input, tf.keras.layers.Dense(2, activation = "softmax")(merged_flatten))

        #####################################################

        # 44 parameters
        def __simple_LSTM__(self) -> tf.keras.Model : 
            return self.add_lstm(input_shape = (3, 120, 1), d_LSTM = 1)

        # doesn't really work all well with the dataset log E = 16-16.5 
        # since empty files raise background traces, which get scaled UP
        # 366 parameters
        def __normed_one_layer_conv2d__(self, model) -> None :

            self.add_input(model, shape = (3, 120, 1))
            self.add_norm(model)
            self.add_conv2d(model, filters = 4, kernel_size = 3, strides = 3)
            self.add_output(model, units = 2, activation = "softmax")

        # 92 parameters
        def __one_layer_conv2d__(self, model) -> None :

            self.add_input(model, shape = (3, 120, 1))
            self.add_conv2d(model, filters = 1, kernel_size = 3, strides = 3)
            self.add_output(model, units = 2, activation = "softmax")

        # 338 parameters
        def __one_large_layer_conv2d__(self, model) -> None : 

            self.add_input(model, shape = (3, 120, 1))
            self.add_conv2d(model, filters = 4, kernel_size = (3,1), strides = 3)
            self.add_output(model, units = 2, activation = "softmax")    

        # 1002 parameters
        def __one_layer_downsampling_equal__(self, model) -> None :

            self.add_input(model, shape = (3, 360, 1))
            self.add_conv2d(model, filters = 4, kernel_size = 3, strides = 3)
            self.add_output(model, units = 2, activation = "softmax")

        # 140 parameters
        def __two_layer_conv2d__(self, model) -> None :

            self.add_input(model, shape = (3, 120, 1))
            self.add_conv2d(model, filters = 4, kernel_size = 3, strides = 3)
            self.add_conv1d(model, filters = 2, kernel_size = 2, strides = 2)
            self.add_output(model, units = 2, activation = "softmax")

        # 630 parameters
        def __two_layer_downsampling_equal__(self, model) -> None :

            self.add_input(model, shape = (3, 360, 1))
            self.add_conv2d(model, filters = 8, kernel_size = 3, strides = 3)
            self.add_conv1d(model, filters = 4, kernel_size = 2, strides = 2)
            self.add_output(model, units = 2, activation = "softmax") 

        # 35 parameters
        def __minimal_conv2d__(self, model) -> None :

            self.add_input(model, shape = (3, 120,1))
            self.add_conv2d(model, filters = 2, kernel_size = (3,2), strides = 2)
            self.add_conv1d(model, filters = 1, kernel_size = 2, strides = 2)
            self.add_conv1d(model, filters = 1, kernel_size = 3, strides = 3)
            self.add_conv1d(model, filters = 1, kernel_size = 3, strides = 3)
            self.add_output(model, units = 2, activation = "softmax")

        # 606 parameters
        def __large_conv2d__(self, model) -> None : 

            self.add_input(model, shape = (3, 120,1))
            self.add_conv2d(model, filters = 4, kernel_size = (3,1), strides = 2)
            self.add_conv1d(model, filters = 8, kernel_size = 3, strides = 3)
            self.add_conv1d(model, filters = 16, kernel_size = 3, strides = 3)
            self.add_conv1d(model, filters = 4, kernel_size = 3, strides = 3)
            self.add_output(model, units = 2, activation = "softmax")

    models = \
        {
            "one_layer_downsampling_equal" : Architectures.__one_layer_downsampling_equal__,
            "two_layer_downsampling_equal" : Architectures.__two_layer_downsampling_equal__,
            "normed_one_layer_conv2d" : Architectures.__normed_one_layer_conv2d__,
            "one_layer_conv2d" : Architectures.__one_layer_conv2d__,
            "one_large_layer_conv2d" : Architectures.__one_large_layer_conv2d__,
            "two_layer_conv2d" : Architectures.__two_layer_conv2d__,
            "minimal_conv2d" : Architectures.__minimal_conv2d__,
            "large_conv2d" : Architectures.__large_conv2d__,
            "simple_LSTM" : Architectures.__simple_LSTM__,
        }

    def __init__(self, name : str, set_architecture = None, supress_print : bool = False) -> None :

        r'''
        :name ``str``: specifies the name of the NN and directory in /cr/data01/filip/models/
        :set_architecture ``str``: specifies the architecture (upon first initialization)

        :Keyword arguments:
        
        * *early_stopping_patience* (``int``) -- number of batches without improvement before training is stopped
        '''

        super().__init__(name)

        if set_architecture is None:

            try:
                self.model = tf.keras.models.load_model("/cr/data01/filip/models/" + name + "/model_converged")
                self.epochs = "converged"
            except OSError:
                available_models = os.listdir('/cr/data01/filip/models/' + name)
                if len(available_models) != 1:
                    choice = input(f"\nSelect epoch from {available_models}\n Model: ")
                else: choice = available_models[0]
                self.model = tf.keras.models.load_model("/cr/data01/filip/models/" + name + "/" + choice)
                self.epochs = int(choice.split("_")[-1])
        else:
            try:
                self.model = tf.keras.Sequential()
                self.models[set_architecture](self.Architectures, self.model)
                self.model.build()
                
            # ValueError is raised for LSTM due to different initialization
            except TypeError:
                self.model = self.models[set_architecture](self.Architectures)

            self.epochs = 0

        self.model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'], run_eagerly = True)

        not supress_print and print(self)

    def train(self, Datasets : tuple, epochs : int, **kwargs) -> None :
        
        # stop if no improvement over 75% of epoch and accuracy over 95%
        early_stopping_patience = kwargs.get("early_stopping_patience", int(0.75 * len(Datasets[0])))
        early_stopping_accuracy = kwargs.get("early_stopping_accuracy", GLOBAL.early_stopping_accuracy)
        
        EarlyStopping = self.BatchwiseEarlyStopping(early_stopping_patience, early_stopping_accuracy)

        callbacks = [EarlyStopping,]

        try:
            os.mkdir(f"/cr/data01/filip/models/{self.name}")
        except FileExistsError: pass

        training_status = "normally"
        TrainingSet, ValidationSet = Datasets

        # print("Creating physics information for both datasets...")
        # TrainingSet.physics_test(n_showers = int(0.05 * TrainingSet.__len__()), save_dir = f"/cr/data01/filip/models/{self.name}/training_set_physics_test.png")
        # ValidationSet.physics_test(n_showers = int(0.2 * ValidationSet.__len__()), save_dir = f"/cr/data01/filip/models/{self.name}/validation_set_physics_test.png")

        try:
            self.model.fit(TrainingSet, validation_data = ValidationSet, epochs = epochs - self.epochs, callbacks = callbacks)
            self.epochs = epochs

        except EarlyStoppingError: 
            self.epochs = "converged"
            training_status = "early"

        self.save()

        with open(f"/cr/data01/filip/models/{self.name}/model_{self.epochs}/training_files.csv", "w") as training_file:
            for file in TrainingSet.files:
                training_file.write(file + "\n")

        with open(f"/cr/data01/filip/models/{self.name}/model_{self.epochs}/validation_files.csv", "w") as validation_file:
            for file in ValidationSet.files:
                validation_file.write(file + "\n")

        # provide some metadata
        print(f"\nTraining exited {training_status}. Onto providing metadata now...")
        self.make_signal_dataset(ValidationSet, f"validation_data")

        # # calculate trigger rate on random traces
        # print(f"\ncalculating trigger rate on 0.5s (~30 000 Hz) of random traces now")
        # f, df, n, n_trig, t = self.production_test(30000)

        # with open(f"/cr/data01/filip/models/{self.name}/model_{self.epochs}/production_test.csv", "w") as random_file:
        #     random_file.write("# f  df  n_traces  n_total_triggered  total_trace_duration\n")
        #     random_file.write(f"{f} {df} {n} {n_trig} {t}")


    def save(self) -> None : 
        self.model.save(f"/cr/data01/filip/models/{self.name}/model_{self.epochs}")

    def __call__(self, signal : np.ndarray) -> typing.Union[bool, tuple] :

        # 1 if the network thinks it's seeing a signal
        # 0 if the network thinks it's seening background 

        if len(signal.shape) == 3:                                                  # predict on batch
            predictions = self.model.predict_on_batch(signal)

            return np.array([prediction.argmax() for prediction in predictions])

        elif len(signal.shape) == 2:                                                # predict on sample
            
            return np.array(self.model( tf.expand_dims([signal], axis = -1) )).argmax()        

    def __str__(self) -> str :
        self.model.summary()
        return ""

    # add a layer to the model architecture
    def add(self, layer : str, **kwargs) -> None :
        print(self.layers[layer], layer, kwargs)
        self.layers[layer](**kwargs)

    # get validation/training file path in a list
    def get_files(self, dataset : str) -> typing.Union[list, None] : 

        if dataset not in ["training", "validation"]:
            print(f"[WARN] -- Attempt to fetch invalid dataset: <{dataset}>")
            return None
        
        return list(np.loadtxt(f"/cr/data01/filip/models/{self.name}/model_{self.epochs}/{dataset}_files.csv", dtype = str))

# Class for streamlined handling of multiple NNs with the same architecture
class Ensemble(NNClassifier):

    def __init__(self, name : str, set_architecture : str = None, n_models : int = GLOBAL.n_ensembles) -> None :

        r'''
        :name ``str``: specifies the name of the NN and directory in /cr/data01/filip/models/ENSEMBLES/
        :set_architecture ``str``: specifies the architecture (upon first initialization)

        :Keyword arguments:
        
        * *early_stopping_patience* (``int``) -- number of batches without improvement before training is stopped
        '''

        try:
            os.mkdir(f"/cr/data01/filip/models/ENSEMBLES/{name}")
        except FileExistsError: pass

        supress_print = False
        self.models = []

        for i in range(1, n_models + 1):
            ThisModel = NNClassifier("ENSEMBLES/" + name + f"/ensemble_{str(i).zfill(2)}/", set_architecture, supress_print)
            self.models.append(ThisModel)

            supress_print = True

        self.name = "ENSEMBLE_" + name

        print(f"{self.name}: {n_models} models successfully initiated\n")

    def train(self, Datasets : tuple, epochs : int, **kwargs) -> None:

        start = perf_counter_ns()

        for i, instance in enumerate(self.models,1):

            time_spent = (perf_counter_ns() - start) * 1e-9
            elapsed = strftime('%H:%M:%S', gmtime(time_spent))
            eta = strftime('%H:%M:%S', gmtime(time_spent * (len(self.models)/i - 1)))

            print(f"Model {i}/{len(self.models)}, {elapsed} elapsed, ETA = {eta}")

            instance.train(Datasets, epochs)

            Datasets[0].__reset__()
            Datasets[1].__reset__()

    def save(self) -> None : 
        for model in self.models: model.save()

    def __call__(self, trace : np.ndarray) -> list :

        return [model(trace) for model in self.models]

    def load_and_print_performance(self, dataset : str, quiet = False) -> list : 

        # TP, FP, TN, FN 
        predictions = []

        # keep the iterable index in case of early breaking during debugging
        for i, model in enumerate(self.models):
            prediction = model.load_and_print_performance(dataset, quiet = quiet)
            predictions.append(prediction)

        return predictions

    def get_background_rates(self) -> tuple : 

        rate, rate_err = [], []

        for model in self.models:
            f,  df = np.loadtxt("/cr/data01/filip/models/" + model.name + "model_" + str(model.epochs) + "/production_test.csv", usecols = [0, 1])
            rate.append(f), rate_err.append(df)

        return rate, rate_err
    
    def get_accuracy(self, dataset : str) -> tuple : 

        acc, acc_err = [], []

        for prediction in self.load_and_print_performance(dataset, quiet = True):
            TP, FP, TN, FN = prediction
            x, o = float(len(TP)), float(len(FN))
            accuracy = x / (x + o)
            err = 1/(x+0)**2 * np.sqrt( x**3 + o**3 - 2 * np.sqrt((x * o)**3) )

            acc.append(accuracy)
            acc_err.append(err)

        return acc, acc_err

    def money_plot(self, *args, **kwargs):

        color = kwargs.get("color", "steelblue")
        fig, ax = plt.subplots()

        ax.set_yscale("log")
        ax.set_ylabel("Random trace trigger rate / $\mathrm{Hz}$")
        ax.set_xlabel("Trigger efficiency")
        ax.set_xlim(0, 1.05)

        HardwareClassifier.plot_performance(ax)

        for i, dataset in enumerate(args):

            rate, rate_err = self.get_background_rates()
            acc, acc_err = self.get_accuracy(dataset)

            n_rate, bins = np.histogram(rate, bins = 10)
            current_y, scaling = min(rate), 0.01
            bin_width = bins[1] - bins[0]
            boxes = []

            for i, rate_bin in enumerate(n_rate):
                boxes.append(Rectangle((np.mean(acc) - scaling * rate_bin, current_y), width = 2*scaling*rate_bin, height = bin_width))
                current_y += bin_width
                    
            ax.add_collection(PatchCollection(boxes, facecolor = "slategray", lw = 0, alpha = 0.2))
            plt.errorbar(np.mean(acc), np.mean(rate), xerr = np.std(acc), yerr = np.std(rate), fmt = "o", markersize = 12, c = color)
            plt.scatter(np.mean(acc), np.mean(rate) + np.std(rate), marker = "1", s = 400, zorder = 4, c = color)
            plt.scatter(np.mean(acc), np.mean(rate) - np.std(rate), marker = "2", s = 400, zorder = 4, c = color)











    def ROC(self, dataset : str, **kwargs : dict) -> None :

        raise NotImplementedError

        warning_orange = '\033[93m'
        end_color_code = '\033[0m'

        predictions = self.load_and_print_performance(dataset)
        best_prediction, worst_prediction = None, None
        best_score, worst_score = -np.inf, np.inf
        all_predictions = [[] for i in range(2)]

        for i, prediction in enumerate(predictions): 

            TP, FP = prediction[0], prediction[1]

            # compare TPR across models
            current_score = ( len(TP) ) / ( len(TP) + len(FP) )

            if current_score > best_score:
                best_prediction = predictions[i]
                best_score = current_score
            if current_score < worst_score:
                worst_prediction = predictions[i]
                worst_score = current_score

            all_predictions[0] += list(TP)
            all_predictions[1] += list(FP)

        all_score = ( len(all_predictions[0]) ) / ( len(all_predictions[0]) + len(all_predictions[1]) )

        print(warning_orange + f"Overall performance (TPR): WORST = {worst_score * 100:.4f}%, BEST = {best_score * 100:.4f}%, ALL = {all_score * 100:.4f}%" + end_color_code)

        # draw the ROC curve for each of the (best, worst, avg.) predictions
        for i, prediction in enumerate([worst_prediction, best_prediction, all_predictions]):

            TP, FP = prediction[0], prediction[1]

            x = np.clip(FP, a_min = 1e-6, a_max = None)
            y = np.clip(TP, a_min = 1e-6, a_max = None)

            score_low, score_high = min([x.min(), y.min()]), max([x.max(), y.max()])
            last, current_x, current_y = score_low, 0, 0
            ROC_x, ROC_y = [],[]

            bins = np.geomspace(score_low, score_high, kwargs.get("n", 50))[::-1]
            norm = ( len(x) + len(y) ) * 5              # why x5 ???

            for score_bin in bins:

                this_x = ((last >= x) & (x > score_bin)).sum()
                this_y = ((last >= y) & (y > score_bin)).sum()
                
                current_x += this_x / norm
                current_y += this_y / norm
                
                ROC_y.append(current_y), ROC_x.append(current_x)

                last = score_bin
            
            ROC_x.append(1), ROC_y.append(1)

            plt.plot(ROC_x, ROC_y, lw = 2, alpha = 1 if i == 1 else 0.4, ls = "solid" if i == 1 else ":", c = kwargs.get("c", "steelblue"))

        plt.plot([],[], label = kwargs.get("label", f"{self.name} - {dataset}"), c = kwargs.get("c", "steelblue"))

        plt.xlim(-0.02,1.02)
        plt.ylim(-0.02,1.02)
        plt.plot([0,1],[0,1], ls = "--", c = "gray")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")

    def PRC(self, dataset : str) -> None :

        raise NotImplementedError

        try:
            if header_was_called: pass

        except NameError:
            self.__header__()

        for i, model in enumerate(self.models):

            model.PRC(dataset, title = f"{self.name}", label = f"model instance {i}")

    def make_signal_dataset(self, Dataset: "Generator", save_dir: str, n_showers: int = None, save_traces: bool = False) -> None:
        
        for model in self.models:
            model.make_signal_dataset(Dataset, save_dir, n_showers, save_traces)

# Wrapper for currently employed station-level triggers (Th1, Th2, ToT, etc.)
# Information on magic numbers comes from Davids Mail on 10.03.22 @ 12:30pm
class HardwareClassifier(Classifier):

    def __init__(self, triggers : list = ["th2", "tot", "totd"], name : str = False) : 
        super().__init__(name or "HardwareClassifier")
        self.triggers = []

        if "th1" in triggers: self.triggers.append(self.Th1)
        if "th2" in triggers: self.triggers.append(self.Th2)
        if "tot" in triggers: self.triggers.append(self.ToT)
        if "totd" in triggers: self.triggers.append(self.ToTd)
        if "mops" in triggers: self.triggers.append(self.MoPS_compatibility)

    def __call__(self, trace : typing.Union[np.ndarray, Trace]) -> bool :

        if isinstance(trace, Trace):

            # check Threshold trigger first, they can receive the entire trace at once
            if self.Th2 in self.triggers or self.Th1 in self.triggers:
                full_trace = np.array([trace.pmt_1, trace.pmt_2, trace.pmt_3])
                
                if self.Th1 in self.triggers and self.Th1(full_trace): return True
                if self.Th2 in self.triggers and self.Th2(full_trace): return True

            # check for rest of the triggers in sliding window analysis
            rest_of_the_triggers = [trigger for trigger in self.triggers if trigger not in [self.Th1, self.Th2]]

            if rest_of_the_triggers == []: return False

            for window in trace:
                for trigger in rest_of_the_triggers:
                    if trigger(window): return True
            else: return False
            
        
        elif isinstance(trace, np.ndarray):
            try:
                for trigger in self.triggers:
                    if trigger(trace): return True
                else: return False

            except ValueError:
                return np.array([self.__call__(t) for t in trace])

    def Th1(self, signal) -> bool : 
        return self.Th(signal, 1.7)

    def Th2(self, signal) -> bool : 
        return self.Th(signal, 3.2)

    # method to check for (coincident) absolute signal threshold
    @staticmethod
    def Th(signal : np.ndarray, threshold : float) -> bool : 

        # Threshold of 3.2 immediately gets promoted to T2
        # Threshold of 1.75 if a T3 has already been issued
        pmt_1, pmt_2, pmt_3 = signal

        # hierarchy doesn't (shouldn't?) matter
        for i in range(signal.shape[1]):
            if pmt_1[i] > threshold:
                if pmt_2[i] > threshold:
                    if pmt_3[i] > threshold:
                        return True
                    else: continue
                else: continue
            else: continue
        
        return False

    # method to check for elevated baseline threshold trigger
    @staticmethod
    def ToT(signal : np.ndarray) -> bool : 

        threshold     = 0.2                                                         # bins above this threshold are 'active'
        n_bins        = 13                                                          # minimum number of bins above threshold
        multiplicity  = 2                                                           # number of PMTs that satisfy conditions

        pmt_1, pmt_2, pmt_3 = signal

        # count initial active bins
        pmt1_active = list(pmt_1 > threshold).count(True)
        pmt2_active = list(pmt_2 > threshold).count(True)
        pmt3_active = list(pmt_3 > threshold).count(True)
        ToT_trigger = [pmt1_active >= n_bins, pmt2_active >= n_bins, pmt3_active >= n_bins]

        if ToT_trigger.count(True) >= multiplicity:
            return True
        else:
            return False

    # method to check for elevated baseline of deconvoluted signal
    # note that this only ever gets applied to UB-like traces, with 25 ns binning
    @staticmethod
    def ToTd(signal : np.ndarray) -> bool : 

        # for information on this see GAP note 2018-01
        dt      = 25                                                                # UB bin width
        tau     = 67                                                                # decay constant
        decay   = np.exp(-dt/tau)                                                   # decay term
        deconvoluted_trace = []

        for pmt in signal:
            deconvoluted_pmt = [(pmt[i] - pmt[i-1] * decay)/(1 - decay) for i in range(1,len(pmt))]
            deconvoluted_trace.append(deconvoluted_pmt)
 
        return HardwareClassifier.ToT(np.array(deconvoluted_trace))
    
    # count the number of rising flanks in a trace
    # WARNING: THIS IMPLEMENTATION IS NOT CORRECT!
    @staticmethod
    def MoPS_compatibility(signal : np.ndarray) -> bool : 

        # assumes we're fed 120 bin trace windows
        y_min, y_max = 3, 31                                                        # only accept y_rise in this range
        min_slope_length = 2                                                        # minimum number of increasing bins
        min_occupancy = 4                                                           # positive steps per PMT for trigger
        pmt_multiplicity = 2                                                        # required multiplicity for PMTs
        pmt_active_counter = 0                                                      # actual multiplicity for PMTs
        integral_check = 75 * 120/250                                               # integral must pass this threshold


        for pmt in signal:

            # check for the modified integral first, computationally easier
            if sum(pmt) * GLOBAL.q_peak <= integral_check: continue
            
            occupancy = 0
            positive_steps = np.nonzero(np.diff(pmt) > 0)[0]
            steps_isolated = np.split(positive_steps, np.where(np.diff(positive_steps) != 1)[0] + 1)
            candidate_flanks = [step for step in steps_isolated if len(step) >= min_slope_length]
            candidate_flanks = [np.append(flank, flank[-1] + 1) for flank in candidate_flanks]

            for i, flank in enumerate(candidate_flanks):

                # adjust searching area after encountering a positive flank
                total_y_rise = (pmt[flank[-1]] - pmt[flank[0]]) * GLOBAL.q_peak
                n_continue_at_index = flank[-1] + max(0, int(np.log2(total_y_rise) - 2))

                for j, consecutive_flank in enumerate(candidate_flanks[i + 1:], 0):

                    if n_continue_at_index <= consecutive_flank[0]: break                                                       # no overlap, no need to do anything
                    elif consecutive_flank[0] < n_continue_at_index <= consecutive_flank[-1]:                                   # partial overlap, adjust next flank
                        overlap_index = np.argmin(np.abs(consecutive_flank - n_continue_at_index))
                        candidate_flanks[i + j] = candidate_flanks[i + j][overlap_index:]
                        if len(candidate_flanks[i + j]) < min_slope_length: _ = candidate_flanks.pop(i + j)
                        break

                    elif consecutive_flank[-1] < n_continue_at_index:                                                           # complete overlap, discard next flank
                        _ = candidate_flanks.pop(i + j)

                if total_y_rise > y_min and total_y_rise < y_max: occupancy += 1
            if occupancy > min_occupancy: pmt_active_counter += 1
            
        return pmt_active_counter >= pmt_multiplicity

    @staticmethod
    def plot_performance(ax : plt.axes, x : float = None, y : float = None, xerr : float = None, yerr : float = None) -> None :
        
        if x is None:                                                                                                           # default dataset: 'q_peak_compatibility' 
            TP, FN = 66548., 100822.
            x = TP / (TP + FN)
            xerr = 1/(TP+FN)**2 * np.sqrt( TP**3 + FN**3 - 2 * np.sqrt((TP * FN)**3) )
        if y is None:                                                                                                           # RunProductionTest/plot_everything.ipynb
            y = 21.3
            yerr = 3

        ax.errorbar(x, y, xerr = xerr, yerr = yerr, c = "k", label = "Classical triggers", fmt = "*")

class BayesianClassifier(Classifier):
    
    def __init__(self) -> None :

        super().__init__("BayesianClassifier")

        self.bin_centers = np.loadtxt("/cr/data01/filip/models/naive_bayes_classifier/bins.csv")
        self.signal = np.loadtxt("/cr/data01/filip/models/naive_bayes_classifier/signal.csv")
        self.background = np.loadtxt("/cr/data01/filip/models/naive_bayes_classifier/background.csv")
        self.quotient = self.signal / self.background
    #     self.threshold = threshold

    def __call__(self, trace : np.ndarray) -> bool : 
        
        # mock_likelihood = 0

        # for PMT in trace:

        #     bins = [self.find_bin(value) for value in PMT]
        #     mock_likelihood += np.sum(np.log(self.quotient[bins]))

        # return mock_likelihood > self.threshold

        return Trace.integrate(trace) > 1.5874890619840887

    # return the index of the bin (from self.bin_centers) that value would fall into
    def find_bin(self, value : int) -> int : 
        return np.abs(self.bin_centers - value).argmin()