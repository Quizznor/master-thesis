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
                LDF = lambda x : lateral_distribution_function(x, LDF_efficiency, LDF_prob_50, 1/LDF_scale)

                c = colormap(t / (len(theta_bins) - 1))
                all_data = hits + misses
                n_data_in_bins = int(20 * np.sqrt(e + 1))

                # have at least 7 bins or bins with >50 samples
                while True:
                    n_bins = len(all_data) // n_data_in_bins
                    probabilities = np.linspace(0, 1, n_bins)
                    binning = mquantiles(all_data, prob = probabilities)
                    bin_center = 0.5 * (binning[1:] + binning[:-1])
                    n_all, _ = np.histogram(all_data, bins = binning)

                    if len(n_all) <= 7: 
                        n_data_in_bins -= 10
                        if n_data_in_bins <= 50: break
                    else: break

                x, _ = np.histogram(hits, bins = binning)
                o, _ = np.histogram(misses, bins = binning)
                efficiency = x / (x + o) * LDF(bin_center)
                efficiency_err = 1/n_all**2 * np.sqrt( x**3 + o**3 - 2 * np.sqrt((x * o)**3) )          # lack LDF error part here !!
                # efficiency_err[efficiency_err == 0] = 1e-3                                              # such that residuals are finite

                filter = np.isnan(efficiency)
                bin_center = bin_center[~filter]
                efficiency = efficiency[~filter]
                efficiency_err = efficiency_err[~filter]

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
                                                                    p0 = [1, bin_center[np.argmin(abs(efficiency - 0.5))], 1/LDF_scale],
                                                                    bounds = ([0, 0, 0], [np.inf, np.inf, 1000]),
                                                                    # sigma = efficiency_err,
                                                                    # absolute_sigma = True,
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

    # plot the classifiers efficiency in terms of deposited signal    
    def signal_efficiency(self, dataset : str, **kwargs) -> None :

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
            TN = np.loadtxt(save_files['TN'], usecols = [8])
        else: TN = np.array([])

        if os.stat(save_files['FP']).st_size:
            FP = np.loadtxt(save_files['FP'], usecols = [8])
        else: FP = np.array([])

        if os.stat(save_files['TP']).st_size:
            TP = np.loadtxt(save_files['TP'], usecols = [8])
        else: TP = np.array([])
        
        if os.stat(save_files['FN']).st_size:
            FN = np.loadtxt(save_files['FN'], usecols = [8])
        else: FN = np.array([])

        signal_bins = np.geomspace(5e-2, 1e3, kwargs.get("n_bins", 100))

        n_hit, _ = np.histogram(TP, bins = signal_bins)
        n_miss, _ = np.histogram(FN, bins = signal_bins)      

        plt.scatter(0.5 * (signal_bins[1:] + signal_bins[:-1]), n_hit / (n_hit + n_miss), c = kwargs.get("color", "steelblue"), label = kwargs.get("label", None))

    # plot the classifiers efficiency in terms of primary energy
    def energy_efficiency(self, dataset : str, angle : float = 38, tolerance : float = 2, **kwargs) -> None : 

        bar_kwargs = \
            {
                "fmt" : kwargs.get("marker", "o"),
                "c" : kwargs.get("color", "steelblue"),
                "elinewidth" : 0.9,
                "capsize" : 4,
                "label" : kwargs.get("label", None)
            }

        e_bins = np.geomspace(10**16, 10**19.5, kwargs.get("n_points", 10))
        TP, FP, TN, FN = self.load_and_print_performance(dataset)

        energy_bins = [10**16, 10**16.5, 10**17, 10**17.5, 10**18, 10**18.5, 10**19, 10**19.5]      # uniform in log(E)
        theta_bins =  [0.0000, 33.5600, 44.4200, 51.3200, 56.2500, 65.3700]                         # pseudo-uniform in sec(θ)

        TP_spd_selected = TP[np.where(TP[:, 0] <= kwargs.get("array_spacing", 1500))]
        TP_selected = TP_spd_selected[np.where(TP_spd_selected[:,2] < angle + tolerance)]
        TP_selected = TP_selected[np.where(angle - tolerance <= TP_selected[:,2])]
        FN_spd_selected = FN[np.where(FN[:, 0] <= kwargs.get("array_spacing", 1500))]
        FN_selected = FN_spd_selected[np.where(FN_spd_selected[:,2] < angle + tolerance)]
        FN_selected = FN_selected[np.where(angle - tolerance <= FN_selected[:,2])]

        indices_to_delete = []
        FN_selected = list(FN_selected)

        # take LDF particle probability into account
        for i, prediction in enumerate(TP_selected):
            spd, energy, theta = prediction[:3]
            eb = np.digitize(energy, energy_bins)
            tb = np.digitize(theta, theta_bins)
            _, (eff, prob50, scale) = get_fit_function("/cr/tempdata01/filip/QGSJET-II/LDF/", eb, tb)
            LDF = lambda x : lateral_distribution_function(x, eff, prob50, 1/scale)

            if np.random.uniform() <= 1 - LDF(spd):
                FN_selected.append(prediction)
                indices_to_delete.append(i)

        FN_selected = np.array(FN_selected)
        TP_selected = np.delete(TP_selected, indices_to_delete, axis = 0)

        hits = [0 for _ in range(len(e_bins) - 1)]
        miss = [0 for _ in range(len(e_bins) - 1)]

        # Sort predictions into bins of energy
        for source, target in zip([TP_selected, FN_selected], [hits, miss]):

            energy_sorted = np.digitize(source[:,1], e_bins)
            values, counts = np.unique(energy_sorted, return_counts = True)

            for (value, count) in zip(values, counts):
                target[value - 1] += count

        hits, miss = np.array(hits), np.array(miss)
        efficiency = hits / (hits + miss)
        eff_err = 1/(hits + miss)**2 * np.sqrt( hits**3 + miss**3 - 2 * np.sqrt((hits * miss)**3) )

        upper = np.clip(efficiency + eff_err, 0, 1)
        lower = np.clip(efficiency - eff_err, 0, 1)

        plt.xscale("log")
        plt.errorbar(0.5 * (e_bins[1:] + e_bins[:-1]), efficiency, yerr = [efficiency - lower, upper - efficiency], **bar_kwargs)
        plt.legend(title = fr"${angle - tolerance}^\circ \leq \theta < {angle + tolerance}^\circ$")

    # relate single station efficiency function to T3 efficiency
    def do_t3_simulation(self, dataset : str, n_points : int = 1e5, **kwargs) -> None :

        import seaborn as sns

        if isinstance(n_points, float): n_points = int(n_points)

        if hasattr(self, "epochs"):
            root_path = f"/cr/data01/filip/models/{self.name}/model_{self.epochs}/ROC_curve/{dataset}/"
        else:
            root_path = f"/cr/data01/filip/models/{self.name}/ROC_curve/{dataset}/"

        colormap = cmap.get_cmap("plasma")
        fig, ax = plt.subplots()

        dx = 100
        theta = kwargs.get("theta", 38)
        tolerance = kwargs.get("tolerance", 4)
        efficiencies = []

        # set up plot
        ax.text( 635, -55, "T3 detected", fontsize = 22)
        ax.text(1395 + dx, 775, "T3 missed", fontsize = 22)
        symmetry_line = lambda x : 1500 - x
        X = np.linspace(700, 1550, 100)
        ax.scatter([0, 1500, 750], [0, 0, 750], s = 100, c = "k")
        ax.scatter([1500 + dx, 750 + dx, 2250 + dx], [0, 750, 750], s = 100, c = "k")
        ax.plot(X + dx/2, symmetry_line(X), ls = "solid", c = "k", zorder = 0, lw = 2)
        ax.add_patch(Polygon([[0,0], [1500, 0], [750, 750]], closed = True, color = "green", alpha = 0.1, lw = 0))
        ax.add_patch(Polygon([[750 + dx, 750], [1500 + dx, 0], [2250 + dx, 750]], closed = True, color = "red", alpha = 0.1, lw = 0))

        # create shower cores in target area
        theta_bins = [0, 26, 38, 49, 60, 90]
        ys = np.random.uniform(0, 750, n_points)
        xs = np.random.uniform(0, 1500, n_points) + ys
        reflect = [ ys[i] > symmetry_line(xs[i]) for i in range(len(xs))]
        xs[reflect] = -xs[reflect] + 2250
        ys[reflect] = -ys[reflect] + 750

        energy_hits, energy_miss = [], []

        energy_bins = [10**16, 10**16.5, 10**17, 10**17.5, 10**18, 10**18.5, 10**19, 10**19.5]      # uniform in log(E)
        theta_bins =  [0.0000, 33.5600, 44.4200, 51.3200, 56.2500, 65.3700]                         # pseudo-uniform in sec(θ)

        start_time = perf_counter_ns()

        # do the T3 simulation
        t3_hits = [[0 for t in range(len(theta_bins) - 1)] for e in range(len(energy_bins) - 1)]
        t3_misses = [[0 for t in range(len(theta_bins) - 1)] for e in range(len(energy_bins) - 1)]
        x_container = [[[] for t in range(len(theta_bins) - 1)] for e in range(len(energy_bins) - 1)]
        y_container = [[[] for t in range(len(theta_bins) - 1)] for e in range(len(energy_bins) - 1)]
        container_hits = [[] for t in range(len(theta_bins) - 1)]
        container_miss = [[] for t in range(len(theta_bins) - 1)]
        stations = [[0, 0, 0], [1500, 0, 0], [750, 750, 0]]

        for step_count, (x, y) in enumerate(zip(xs, ys)):

            progress_bar(step_count, n_points, start_time)

            random_phi = np.random.uniform(0, 360)
            random_energy = 10**(np.random.uniform(low = 16, high = 19.5))
            random_zenith = np.random.uniform(low = 0, high = 65)
            eb = np.digitize(random_energy, energy_bins) - 1
            tb = np.digitize(random_zenith, theta_bins) - 1

            fit_function, _ = get_fit_function(root_path, eb, tb)

            sp_distances = []
            
            for station in stations:

                core_position = np.array([x, y, 0])
                core_origin = np.sin(random_zenith) * np.array([np.cos(random_phi), np.sin(random_phi), 1/np.tan(random_zenith)]) + core_position

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
                t3_hits[eb][tb] += 1
                container_hits[tb].append(random_energy)

            else:
                x, y = 2250 - x + dx, 750 - y
                t3_misses[eb][tb] += 1
                container_miss[tb].append(random_energy)

                if theta - tolerance/2 <= random_zenith < theta + tolerance/2:
                    energy_miss.append(random_energy)

            x_container[eb][tb].append(x)
            y_container[eb][tb].append(y)

        size_bins = [30, 50, 70, 90, 110, 160, 200]
        e_labels = [r"$16$", r"$16.5$", r"$17$", r"$17.5$", r"$18$", r"$18.5$", r"$19$", r"$19.5$"]

        for e, (x_energy, y_energy) in enumerate(zip(x_container, y_container)):
            for t, (x, y) in enumerate(zip(x_energy, y_energy)):

                c = colormap(t / len(x_energy))
                s = size_bins[e]

                ax.scatter(x[::kwargs.get("step", 50)], y[::kwargs.get("step", 50)], color = c, s = s, marker = "x")

        for e, bin in enumerate(size_bins):
            ax.scatter([],[], c = "k", s = bin, label = e_labels[e] + r" $\leq$ log$_{10}$($E$ / $\mathrm{eV}$) $<$ " + e_labels[e + 1], marker = "x")

        ax.set_aspect('equal')
        ax.legend(fontsize = 18)
        plt.xlabel("Easting / m")
        plt.ylabel("Northing / m")

        norm = BoundaryNorm(theta_bins, colormap.N)
        ax2 = fig.add_axes([0.91, 0.2, 0.01, 0.6])

        cbar = ColorbarBase(ax2, cmap=colormap, norm=norm, label = r"sec$(\theta)$ - 1")
        cbar.set_ticks(theta_bins)
        cbar.set_ticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.4"])

        plt.figure()

        theta_bins =  [0.0000, 33.5600, 44.4200, 51.3200, 56.2500, 65.3700]                         # pseudo-uniform in sec(θ)

        e_labels = EventGenerator.libraries.keys()
        t_labels = ["0_34", "34_44", "44_51", "51_56", "56_65"]

        t3_hits = np.array(t3_hits)
        t3_misses = np.array(t3_misses)

        sns.heatmap(t3_hits / (t3_hits + t3_misses) * 1e2, annot = True, fmt = ".1f", cbar_kws = {"label" : "T3 efficiency / \%"})
        plt.xticks(ticks = 0.5 + np.arange(0, 5, 1), labels = t_labels, fontsize = 28)
        plt.yticks(ticks = 0.5 + np.arange(0, 7, 1), labels = e_labels, fontsize = 28)
        plt.xlabel("Zenith range")
        plt.ylabel("Energy range")

        fig = plt.figure()

        e_bins_efficiency = np.geomspace(10**16, 10**19.5, kwargs.get("e_bins", len(energy_bins) - 1))

        for t, (energy_hits, energy_miss) in enumerate(zip(container_hits, container_miss)):
            n_hit, bins = np.histogram(energy_hits, bins = e_bins_efficiency)
            n_miss, bins = np.histogram(energy_miss, bins = e_bins_efficiency)
            eff_err = 1/(n_hit + n_miss)**2 * np.sqrt( n_hit**3 + n_miss**3 - 2 * np.sqrt((n_hit * n_miss)**3) )
            eff = n_hit / (n_miss + n_hit)
            efficiencies.append(eff)

            upper = np.clip(eff + eff_err, 0, 1)
            lower = np.clip(eff - eff_err, 0, 1)

            plt.errorbar(0.5 * (bins[1:] + bins[:-1]), eff, yerr = [eff - lower, upper - eff], fmt = "-o", c = colormap(t / (len(theta_bins) - 1)))

        plt.ylabel("T3 Trigger efficiency")
        plt.xlabel("Primary energy / $\mathrm{eV}$")
        plt.xscale("log")

        norm = BoundaryNorm(theta_bins, colormap.N)
        ax2 = fig.add_axes([0.91, 0.1, 0.01, 0.8])

        cbar = ColorbarBase(ax2, cmap=colormap, norm=norm, label = r"sec$(\theta)$ - 1")
        cbar.set_ticks(theta_bins)
        cbar.set_ticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.4"])

        return 0.5 * (bins[1:] + bins[:-1]), efficiencies

    # calculate trigger efficiency over predictions scaled by energy (flux)
    def get_true_accuracy(self, TP : np.ndarray, FN : np.ndarray) -> float : 

        x, e = np.loadtxt("/cr/users/filip/Binaries/energy_histogram.csv", unpack = True)

        efficiency_scaled = 0
        efficiency_unscaled = 0

        for prediction in TP:
            print(prediction)
            raise StopIteration


        # print("logl")


    @staticmethod
    def __header__() -> None : 

        global header_was_called
        header_was_called = True

        print(f"\n{'Classifier':<45} {'Dataset':<35} {'TP':>7} {'FP':>7} {'TN':>7} {'FN':>7}")


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


class MockClassifier(Classifier):

    def __init__(self): pass

    def __call__(self, trace : typing.Union[Trace, np.ndarray]) -> bool : 
        return True

    # relate single station efficiency function to T3 efficiency
    def do_t3_simulation(self, n_points : int = 1e5, **kwargs) -> tuple :

        import seaborn as sns

        if isinstance(n_points, float): n_points = int(n_points)

        root_path = "/cr/tempdata01/filip/QGSJET-II/LDF/"

        colormap = cmap.get_cmap("plasma")
        fig, ax = plt.subplots()

        dx = 100
        theta = kwargs.get("theta", 38)
        tolerance = kwargs.get("tolerance", 4)
        efficiencies = []

        # set up plot
        ax.text( 635, -55, "T3 detected", fontsize = 22)
        ax.text(1395 + dx, 775, "T3 missed", fontsize = 22)
        symmetry_line = lambda x : 1500 - x
        X = np.linspace(700, 1550, 100)
        ax.scatter([0, 1500, 750], [0, 0, 750], s = 100, c = "k")
        ax.scatter([1500 + dx, 750 + dx, 2250 + dx], [0, 750, 750], s = 100, c = "k")
        ax.plot(X + dx/2, symmetry_line(X), ls = "solid", c = "k", zorder = 0, lw = 2)
        ax.add_patch(Polygon([[0,0], [1500, 0], [750, 750]], closed = True, color = "green", alpha = 0.1, lw = 0))
        ax.add_patch(Polygon([[750 + dx, 750], [1500 + dx, 0], [2250 + dx, 750]], closed = True, color = "red", alpha = 0.1, lw = 0))

        # create shower cores in target area
        theta_bins = [0, 26, 38, 49, 60, 90]
        ys = np.random.uniform(0, 750, n_points)
        xs = np.random.uniform(0, 1500, n_points) + ys
        reflect = [ ys[i] > symmetry_line(xs[i]) for i in range(len(xs))]
        xs[reflect] = -xs[reflect] + 2250
        ys[reflect] = -ys[reflect] + 750

        energy_hits, energy_miss = [], []

        energy_bins = [10**16, 10**16.5, 10**17, 10**17.5, 10**18, 10**18.5, 10**19, 10**19.5]      # uniform in log(E)
        theta_bins =  [0.0000, 33.5600, 44.4200, 51.3200, 56.2500, 65.3700]                         # pseudo-uniform in sec(θ)

        start_time = perf_counter_ns()

        # do the T3 simulation
        t3_hits = [[0 for t in range(len(theta_bins) - 1)] for e in range(len(energy_bins) - 1)]
        t3_misses = [[0 for t in range(len(theta_bins) - 1)] for e in range(len(energy_bins) - 1)]
        x_container = [[[] for t in range(len(theta_bins) - 1)] for e in range(len(energy_bins) - 1)]
        y_container = [[[] for t in range(len(theta_bins) - 1)] for e in range(len(energy_bins) - 1)]
        container_hits = [[] for t in range(len(theta_bins) - 1)]
        container_miss = [[] for t in range(len(theta_bins) - 1)]
        stations = [[0, 0, 0], [1500, 0, 0], [750, 750, 0]]

        for step_count, (x, y) in enumerate(zip(xs, ys)):

            progress_bar(step_count, n_points, start_time)

            random_phi = np.random.uniform(0, 360)
            random_energy = 10**(np.random.uniform(low = 16, high = 19.5))
            random_zenith = np.random.uniform(low = 0, high = 65)
            eb = np.digitize(random_energy, energy_bins) - 1
            tb = np.digitize(random_zenith, theta_bins) - 1

            _, (eff, prob50, scale) = get_fit_function(root_path, eb, tb)
            fit_function = lambda x : lateral_distribution_function(x, eff, prob50, 1/scale)

            sp_distances = []
            
            for station in stations:

                core_position = np.array([x, y, 0])
                core_origin = np.sin(random_zenith) * np.array([np.cos(random_phi), np.sin(random_phi), 1/np.tan(random_zenith)]) + core_position

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
                t3_hits[eb][tb] += 1
                container_hits[tb].append(random_energy)

            else:
                x, y = 2250 - x + dx, 750 - y
                t3_misses[eb][tb] += 1
                container_miss[tb].append(random_energy)

                if theta - tolerance/2 <= random_zenith < theta + tolerance/2:
                    energy_miss.append(random_energy)

            x_container[eb][tb].append(x)
            y_container[eb][tb].append(y)

        size_bins = [30, 50, 70, 90, 110, 160, 200]
        e_labels = [r"$16$", r"$16.5$", r"$17$", r"$17.5$", r"$18$", r"$18.5$", r"$19$", r"$19.5$"]

        for e, (x_energy, y_energy) in enumerate(zip(x_container, y_container)):
            for t, (x, y) in enumerate(zip(x_energy, y_energy)):

                c = colormap(t / len(x_energy))
                s = size_bins[e]

                ax.scatter(x[::kwargs.get("step", 50)], y[::kwargs.get("step", 50)], color = c, s = s, marker = "x")

        for e, bin in enumerate(size_bins):
            ax.scatter([],[], c = "k", s = bin, label = e_labels[e] + r" $\leq$ log$_{10}$($E$ / $\mathrm{eV}$) $<$ " + e_labels[e + 1], marker = "x")

        ax.set_aspect('equal')
        ax.legend(fontsize = 18)
        plt.xlabel("Easting / m")
        plt.ylabel("Northing / m")

        norm = BoundaryNorm(theta_bins, colormap.N)
        ax2 = fig.add_axes([0.91, 0.2, 0.01, 0.6])

        cbar = ColorbarBase(ax2, cmap=colormap, norm=norm, label = r"sec$(\theta)$ - 1")
        cbar.set_ticks(theta_bins)
        cbar.set_ticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.4"])

        plt.figure()

        theta_bins =  [0.0000, 33.5600, 44.4200, 51.3200, 56.2500, 65.3700]                         # pseudo-uniform in sec(θ)

        e_labels = EventGenerator.libraries.keys()
        t_labels = ["0_34", "34_44", "44_51", "51_56", "56_65"]

        t3_hits = np.array(t3_hits)
        t3_misses = np.array(t3_misses)

        sns.heatmap(t3_hits / (t3_hits + t3_misses) * 1e2, annot = True, fmt = ".1f", cbar_kws = {"label" : "T3 efficiency / \%"})
        plt.xticks(ticks = 0.5 + np.arange(0, 5, 1), labels = t_labels, fontsize = 28)
        plt.yticks(ticks = 0.5 + np.arange(0, 7, 1), labels = e_labels, fontsize = 28)
        plt.xlabel("Zenith range")
        plt.ylabel("Energy range")

        fig = plt.figure()

        e_bins_efficiency = np.geomspace(10**16, 10**19.5, kwargs.get("e_bins", len(energy_bins) - 1))

        for t, (energy_hits, energy_miss) in enumerate(zip(container_hits, container_miss)):
            n_hit, bins = np.histogram(energy_hits, bins = e_bins_efficiency)
            n_miss, bins = np.histogram(energy_miss, bins = e_bins_efficiency)
            eff_err = 1/(n_hit + n_miss)**2 * np.sqrt( n_hit**3 + n_miss**3 - 2 * np.sqrt((n_hit * n_miss)**3) )
            eff = n_hit / (n_miss + n_hit)

            efficiencies.append(eff)

            upper = np.clip(eff + eff_err, 0, 1)
            lower = np.clip(eff - eff_err, 0, 1)

            plt.errorbar(0.5 * (bins[1:] + bins[:-1]), eff, yerr = [eff - lower, upper - eff], fmt = "-o", c = colormap(t / (len(theta_bins) - 1)))

        plt.ylabel("T3 Trigger efficiency")
        plt.xlabel("Primary energy / $\mathrm{eV}$")
        plt.xscale("log")

        norm = BoundaryNorm(theta_bins, colormap.N)
        ax2 = fig.add_axes([0.91, 0.1, 0.01, 0.8])

        cbar = ColorbarBase(ax2, cmap=colormap, norm=norm, label = r"sec$(\theta)$ - 1")
        cbar.set_ticks(theta_bins)
        cbar.set_ticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.4"])

        return 0.5 * (bins[1:] + bins[:-1]), efficiencies