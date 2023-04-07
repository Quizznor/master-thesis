from .__config__ import *
from .Signal import *

# Wrapper for the Generator class
class EventGenerator():

    libraries = \
    {
        "16_16.5" : "/cr/tempdata01/filip/QGSJET-II/protons/16_16.5/",
        "16.5_17" : "/cr/tempdata01/filip/QGSJET-II/protons/16.5_17/",
        "17_17.5" : "/cr/tempdata01/filip/QGSJET-II/protons/17_17.5/",
        "17.5_18" : "/cr/tempdata01/filip/QGSJET-II/protons/17.5_18/",
        "18_18.5" : "/cr/tempdata01/filip/QGSJET-II/protons/18_18.5/",
        "18.5_19" : "/cr/tempdata01/filip/QGSJET-II/protons/18.5_19/",
        "19_19.5" : "/cr/tempdata01/filip/QGSJET-II/protons/19_19.5/",
        # "test"    : "/cr/users/filip/Simulation/TestShowers/"
    }

    def __new__(self, datasets : typing.Union[list, str], **kwargs : dict) -> typing.Union[tuple, "EventGenerator"] :

        r'''
        :datasets ``list[str]``: number of libraries you want included. Note that "all" includes everything.

        :Keyword arguments:
        
        __:Generator options:_______________________________________________________

        * *split* (``float``) -- fraction of of training set/entire set
        * *seed* (``bool``) -- fix randomizer seed for reproducibility
        * *prior* (``float``) -- p(signal), p(background) = 1 - prior

        __:VEM traces:______________________________________________________________

        * *apply_downsampling* (``bool``) -- make UUB traces resembel UB traces
        * *random_phase* (``int [0, 1, 2]``) -- the random phase for downsampling
        * *real_background* (``bool``) -- use real background from random traces
        * *random_index* (``int``) -- which file to use first in random traces
        * *q_peak* (``float``) -- ADC to VEM conversion factor, for UB <-> UUB
        * *q_charge* (``float``) -- Conversion factor for the integral trace
        * *n_bins* (``int``) -- generate a baseline with <trace_length> bins
        * *force_inject* (``int``) -- force the injection of <force_inject> background particles
        * *sigma* (``float``) -- baseline std in ADC counts, ignored for real_background
        * *mu* (``list``) -- mean ADC level in ADC counts, ignored for real_background


        __:Classifier:______________________________________________________________

        * *window* (``int``) -- the length of the sliding window
        * *step* (``int``) -- step size of the sliding window analysis
        * *ignore_low_vem* (``float``) -- intentionally mislabel low VEM_charge signals
        * *ignore_particles* (``int``) -- intentionally mislabel few-particle signals

        '''

        # set top-level environmental variables
        split = kwargs.get("split", GLOBAL.split)
        seed = kwargs.get("seed", GLOBAL.seed)

        # both kwargs not needed anymore, throw them away
        for kwarg in ["split", "seed"]:
            try: del kwargs[kwarg]
            except KeyError: pass                                      
        
        # set RNG seed if desired
        if seed:
            random.seed(seed)                                                       # does this perhaps already fix the numpy seeds?
            np.random.seed(seed)                                                    # numpy docs says this is legacy, maybe revisit?

        # get all signal files
        all_files = self.get_signal_files(datasets)

        # split files into training and testing set (if needed)
        if split in [0,1]:
            kwargs["for_training"] = False
            return Generator(all_files, **kwargs)
        else:
            
            split_files_at_index = int(split * len(all_files))
            training_files = all_files[0:split_files_at_index]
            validation_files = all_files[split_files_at_index:-1]

            kwargs["for_training"] = True
            TrainingSet = Generator(training_files, **kwargs)
            TestingSet = Generator(validation_files, **kwargs)

            return TrainingSet, TestingSet
        
    @staticmethod
    # TODO: support splitting for zenith?
    def get_signal_files(user_choice : typing.Union[list[str], str]) -> list[str] :

        libraries = ["16_16.5", "16.5_17", "17_17.5", "17.5_18", "18_18.5", "18.5_19", "19_19.5"]

        if isinstance(user_choice, str):
            if user_choice in libraries: data = [EventGenerator.libraries[user_choice]]                     # add single energy library
            elif ":" in user_choice:                                                                        # support slicing of energies

                data = []
                low, high = user_choice.split(':')
                low = libraries.index(low) if low else 0
                high = libraries.index(high) if high else -1

                for energy in libraries[low : high]:
                        data.append(EventGenerator.libraries[energy])

            elif user_choice == "all": data = [*EventGenerator.libraries.values()]                          # add every available library
            elif os.path.isdir(user_choice): data = [user_choice]                                           # add a custom directory

            else: raise NotImplementedError(f"input {user_choice} is not supported as argument for 'dataset'")

            all_files = [[os.path.abspath(os.path.join(library, p)) for p in os.listdir(library)] for library in data]
            all_files = [item for sublist in all_files for item in sublist if item.endswith(".csv")]

        elif isinstance(user_choice, list):                                                                 # support list input
            all_files = []

            for item in user_choice:
                all_files.append(EventGenerator.get_signal_files(item))

        # check Ensemble first, due to Ensemble.__super__ == NNClassifier evaluating to true
        elif isinstance(user_choice, Ensemble): all_files = user_choice.models[0].get_files("validation")   # add validation files from ensemble            
        elif isinstance(user_choice, NNClassifier): all_files =  user_choice.get_files("validation")        # add validation files from classifier

        else: raise NotImplementedError(f"input {type(user_choice)} is not supported as argument for 'dataset'")
        
        random.shuffle(all_files)
        return all_files

# Actual generator class that generates training data on the fly
# See this website for help on a working example: shorturl.at/fFI09
class Generator(tf.keras.utils.Sequence):

    labels = \
    {
        # easier access to Background label
        0: tf.keras.utils.to_categorical(0, 2, dtype = int),
        "BKG": tf.keras.utils.to_categorical(0, 2, dtype = int),
        "bkg": tf.keras.utils.to_categorical(0, 2, dtype = int), 

        # easier access to Signal label
        1: tf.keras.utils.to_categorical(1, 2, dtype = int),
        "SIG": tf.keras.utils.to_categorical(1, 2, dtype = int),
        "sig": tf.keras.utils.to_categorical(1, 2, dtype = int)
    }

    def __init__(self, signal_files : list, **kwargs : dict) :

        r'''
        :Keyword arguments:
        
        __:Generator options:_______________________________________________________

        * *prior* (``float``)               -- p(signal), p(background) = 1 - prior
        * *sliding_window_length* (``int``) -- length of the sliding window
        * *sliding_window_step* (``int``)   -- stepsize for the sliding window
        * *real_background* (``bool``)      -- use real background from random traces
        * *random_index* (``int``)          -- which file to use first in random traces
        * *force_inject* (``int``)          -- inject <force_inject> background particles
        * *for_training* (``bool``)         -- return labelled batches if *True* 

        __:VEM traces:______________________________________________________________

        * *apply_downsampling* (``bool``)   -- make UUB traces resembel UB traces
        * *q_peak* (``float``)              -- ADC to VEM conversion factor, for ADC <-> VEM
        * *q_charge* (``float``)            -- Conversion factor for the integral trace
        * *n_bins* (``int``)                -- generate a baseline with <trace_length> bins
        * *floor_trace* (``bool``)          -- floor trace before dividing by q_peak/q_charge
        * *sigma* (``float``)               -- standard deviation of gaussian baseline
        * *is_vem* (``bool``)               -- the library contains already calibrated traces

        __:Classifier:______________________________________________________________

        * *ignore_low_vem* (``float``)      -- intentionally mislabel low VEM_charge signals
        * *ignore_particles* (``int``)      -- intentionally mislabel few-particle signals
        * *particle_type* (``list[str]``)   -- what particles to consider in particle cut

        '''
        
        self.all_kwargs = kwargs                                                                            # for copy functionality
        self.for_training = kwargs.get("for_training")                                                      # return traces AND labels
        self.files = signal_files                                                                           # all signal files in lib
        self.__iteration_index = 0                                                                          # to support iteration

        # Trace building options
        self.trace_length = kwargs.get("trace_length", GLOBAL.trace_length)
        self.force_inject = kwargs.get("force_inject", GLOBAL.force_inject)
        self.apply_downsampling = kwargs.get("apply_downsampling", GLOBAL.downsampling)
        self.baseline_std = kwargs.get("sigma", GLOBAL.baseline_std)
        self.q_peak = kwargs.get("q_peak", np.array([GLOBAL.q_peak for _ in range(3)]))
        self.q_charge = kwargs.get("q_charge", np.array([GLOBAL.q_charge for _ in range(3)]))
        self.trace_options = \
        {
            "window_length"         : kwargs.get("window_length", GLOBAL.window),
            "window_step"           : kwargs.get("window_step", GLOBAL.step),
            "floor_trace"           : kwargs.get("floor_trace", GLOBAL.floor_trace),
            "random_phase"          : kwargs.get("random_phase", GLOBAL.random_phase),
            "is_vem"                : kwargs.get("is_vem", GLOBAL.is_vem),
            "apply_downsampling"    : self.apply_downsampling,
            "force_inject"          : self.force_inject,
            "trace_length"          : self.trace_length,
            "simulation_q_charge"   : self.q_charge,
            "simulation_q_peak"     : self.q_peak,
        }

        # Generator options
        self.use_real_background = kwargs.get("real_background", GLOBAL.real_background)                    # use random trace baselines
        random_index = kwargs.get("random_index", GLOBAL.random_index)                                      # start at this random file
        station = kwargs.get("station", GLOBAL.station)                                                     # use this random station
        self.prior = kwargs.get("prior", GLOBAL.prior)                                                      # probability of signal traces

        # Classifier options
        self.ignore_low_VEM = kwargs.get("ignore_low_vem", GLOBAL.ignore_low_VEM)                           # integrated signal cut threshold
        self.ignore_particles = kwargs.get("ignore_particles", GLOBAL.ignore_particles)                     # particle in tank cut threshold
        self.particle_type = kwargs.get("particle_type", GLOBAL.particle_type)                              # type of particles that count

        if isinstance(self.particle_type, str): self.particle_type = [self.particle_type]

        if self.use_real_background:
            self.RandomTraceBuffer = RandomTrace(station = station, index = random_index)

        if self.use_real_background and self.force_inject is None: self.force_inject = 0

        # Build first background trace
        baseline = self.build_baseline()
        self.BackgroundTrace = Trace(baseline, None, self.trace_options)

        # For labelling/prior stuff
        self.__n_sig, self.__n_bkg, self.__n_int, self.__n_prt = np.NaN, np.NaN, np.NaN, np.NaN

    # number of batches in generator
    def __len__(self) -> int : 
        return len(self.files)

    # generator method to create data on runtime during e.g. training or other analysis
    def __getitem__(self, index : int) -> typing.Union[tuple[np.ndarray], np.ndarray] :

        r'''
        * *for_training = True* -- used for trace diagnostics, full traces that stem from the same shower, returns: *Traces*
        * *for_training = False* -- should ONLY be used during training, returns labelled batches, returns *(Traces, Labels)*
        '''

        n_sig, n_int, n_prt, n_bkg = 0, 0, 0, 0                                                                 # reserve space for loop variables

        if self.prior != 0:

            try:
                stations = SignalBatch(self.files[index]) if self.prior != 0 else []                            # load this shower file in memory
                full_traces, traces, labels = [], [], []                                                        # reserve space for return values
            except Exception as e:
                sys.exit(f"{e} reading signal data from {self.files[index]}")

            for station in stations:

                baseline = self.build_baseline()
                VEMTrace = Trace(baseline, station, self.trace_options, self.files[index])                  # create the trace
                full_traces.append(VEMTrace)

                if not self.for_training: continue

                # should ONLY be used during training, need to return labelled trace windows 
                # where population of SIG and BKG conform to the prior set in __init__    
                else:

                    # add signal data to training batch
                    for window in VEMTrace:

                        this_label, _ = self.calculate_label(VEMTrace)
                        
                        if this_label == "PRT":
                            label = "BKG"
                            n_prt += 1
                        elif this_label == "INT":
                            label = "BKG"
                            n_int += 1
                        elif this_label == "SIG":
                            label = this_label
                            n_sig += 1
                        else:
                            label = this_label
                            n_bkg += 1
                            print(f"[WARN] -- invalid {label = } encountered in signal creation!")

                        traces.append(window), labels.append(Generator.labels[label])

            # return full traces for analysis purposes
            if not self.for_training:
                return full_traces
            
            else:
                
                # (traces), (labels) at this moment only contains signal traces
                # fill up with background according to prior set in __init__...
                n_bkg = int(n_sig * (1 - self.prior)/self.prior) - (n_int + n_prt)

                # all traces from shower were rejected, try again...
                # this may cause a RecursionError with super high cuts
                if n_sig == 0:
                    return self.__getitem__(np.random.randint(self.__len__()))

                # more traces than required prior were rejected, delete some "background" traces
                elif n_bkg < 0:
                    while n_bkg < 0:
                        
                        false_backgrounds = np.unique(np.argwhere(labels == self.labels["BKG"])[:, 0])
                        random_delete = np.random.choice(false_backgrounds)
                        _, _ = traces.pop(random_delete), labels.pop(random_delete)

                        n_bkg += 1
                        
                        if self.ignore_particles: n_prt -= 1
                        elif self.ignore_low_VEM: n_int -= 1

                # less traces than required prior were rejected, add some from library
                elif n_bkg > 0:
                    for _ in range(n_bkg):
                        traces.append(self.get_background_window()), labels.append(self.labels["BKG"])

                # by chance exactly right amount of traces were rejected, do nothing
                else:
                    pass

                # # shuffle traces / labels
                # p = np.random.permutation(len(traces))
                # traces, labels = np.array(traces)[p], np.array(labels)[p]
                traces, labels = np.array(traces), np.array(labels)
                self.__n_sig, self.__n_bkg, self.__n_int, self.__n_prt = n_sig, n_bkg, n_int, n_prt

                return traces, labels
        
        # Training with prior = 0 is stupid, hence assume self.for_training = False
        # returns 1 Background trace in __getitem__ if prior is set to zero by user
        else:
            baseline = self.build_baseline()

            # convert this to an np.ndarray to keep continuity of return type
            return np.array([Trace(baseline, None, self.trace_options)])

    # make this class iterable, yields (traces), (labels) iteratively
    def __iter__(self) -> typing.Generator[tuple[np.ndarray], np.ndarray, StopIteration] : 

        while self.__iteration_index < self.__len__():

            yield self.__getitem__(self.__iteration_index)
            self.__iteration_index += 1

        return StopIteration

    # reset the internal state of the generator
    def __reset__(self) -> None : 

        random.shuffle(self.files)
        self.__iteration_index = 0

    # calculate the VEM trace window label given cuts
    def calculate_label(self, VEMTrace : Trace) -> str :

        integral = VEMTrace.integral_window[VEMTrace._iteration_index]

        # Safety check for minimalistic integral calculation
        if np.isnan(integral): raise ValueError

        if not VEMTrace.has_signal: return "BKG", integral
        else:

            # check particles first (computationally easier)
            if self.ignore_particles:
                total_particles = 0
                for key in self.particle_type:
                    total_particles += VEMTrace.particles[key]
                if total_particles <= self.ignore_particles:
                    return "PRT", integral

            # check integral trace next for cut threshold
            if self.ignore_low_VEM:
                if integral <= self.ignore_low_VEM: 
                    return "INT", integral 

            return "SIG", integral

    # make a copy of this generator (same event files) with different keywords
    def copy(self, **kwargs : dict) -> "Generator" :

        new_kwargs = self.all_kwargs
        for kwarg in kwargs:
            new_kwargs[kwarg] = kwargs[kwarg]

        NewGenerator = Generator(self.files, **new_kwargs)

        return NewGenerator

    # generator for background traces used to fill up __getitem__
    def get_background_window(self) -> np.ndarray : 

        self.BackgroundTrace._iteration_index += self.BackgroundTrace.window_step
        trace_length = self.trace_length if not self.apply_downsampling else self.trace_length // 3

        if self.BackgroundTrace._iteration_index >= trace_length - self.BackgroundTrace.window_length:

            baseline = self.build_baseline()
            self.BackgroundTrace = Trace(baseline, None, self.trace_options)
            return self.get_background_window()

        return self.BackgroundTrace.get_trace_window(self.BackgroundTrace._iteration_index - self.BackgroundTrace.window_step)

    # create a baseline according to options set in __init__
    def build_baseline(self) -> np.ndarray :

        if self.use_real_background: 
            q_peak, q_charge, baseline = self.RandomTraceBuffer.get()                                       # load random trace baseline
            baseline += np.random.uniform(0, 1, size = baseline.shape)                                      # convert int ADC to float ADC

            self.trace_options["baseline_q_charge"] = q_charge
            self.trace_options["baseline_q_peak"] = q_peak

        else:
            baseline = Baseline(GLOBAL.baseline_mean, self.baseline_std, self.trace_length)                 # or create mock gauss. baseline

            self.trace_options["baseline_q_charge"] = self.q_charge
            self.trace_options["baseline_q_peak"] = self.q_peak



        return baseline

    # find index of a file in self.files
    def find(self, filename : str) -> int :
        
        if "/" in filename: filename = filename.split('/')[-1]
        truncated_files = [file.split('/')[-1] for file in self.files]
        return truncated_files.index(filename)

    # getter for __getitem__ statistics during iteration
    def get_train_loop_statistics(self) -> tuple :

        n_sig, n_bkg, n_int, n_prt = self.__n_sig, self.__n_bkg, self.__n_int, self.__n_prt

        if np.any(np.isnan([self.__n_sig, self.__n_bkg, self.__n_int, self.__n_prt])): 
            raise ValueError
        
        self.__n_sig, self.__n_bkg, self.__n_int, self.__n_prt = np.NaN, np.NaN, np.NaN, np.NaN

        return n_sig, n_bkg, n_int, n_prt

    # run some diagnostics on physical variables
    def physics_test(self, n_showers : int = None, save_dir : str = None) -> None :

        if n_showers is None: n_showers = self.__len__()
        temp, self.for_training = self.for_training, False
        all_muons, all_electrons, all_photons = [], [], []
        all_energy, all_zenith, all_spd, all_integral = [], [], [], []
        x_sig, x_bkg = [], []

        sel_muons, sel_electrons, sel_photons = [], [], []
        sel_energy, sel_zenith, sel_spd, sel_integral = [], [], [], []
        sel_x_sig, sel_x_bkg = [], []
        
        start_time = perf_counter_ns()

        for batch, Shower in enumerate(self):
            
            progress_bar(batch, n_showers, start_time)
            
            for trace in Shower:

                has_label = False

                # check for cut requirements set in __init__
                if (self.ignore_low_VEM or self.ignore_particles) and trace.has_signal:
                    for _ in trace:
                        if self.calculate_label(trace)[0] == "SIG": 
                            has_label = True
                            break

                max_sig = trace.Signal.max() / np.mean(trace.simulation_q_peak)
                max_bkg = trace.Baseline.max() / np.mean(trace.simulation_q_peak)
                                    
                # Shower metadata
                if trace.has_signal:
                    all_energy.append(trace.Energy)
                    all_zenith.append(trace.Zenith)
                    all_spd.append(trace.SPDistance)
                    all_muons.append(trace.particles["muons"])
                    all_electrons.append(trace.particles["electrons"])
                    all_photons.append(trace.particles["photons"])

                    x_sig.append(max_sig)
                    x_bkg.append(max_bkg)

                all_integral.append(np.mean(trace.deposited_signal))              

                if has_label:
                    sel_energy.append(trace.Energy)
                    sel_zenith.append(trace.Zenith)
                    sel_spd.append(trace.SPDistance)
                    sel_muons.append(trace.particles["muons"])
                    sel_electrons.append(trace.particles["electrons"])
                    sel_photons.append(trace.particles["photons"])
                    sel_integral.append(np.mean(trace.deposited_signal))

                    sel_x_sig.append(max_sig)
                    sel_x_bkg.append(max_bkg)

                # Injection component
                # TODO ...

            if batch == n_showers: break

        _, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(2, 3)

        # Energy distribution
        ax0.set_title("Energy distribution")
        ax0.hist(all_energy, histtype = "step", bins = np.geomspace(10**16, 10**19.5, 100), label = "All showers")
        ax0.hist(sel_energy, histtype = "step", bins = np.geomspace(10**16, 10**19.5, 100), label = "Selected showers")
        ax0.legend(fontsize = 16)

        for e_cut in [10**16, 10**16.5, 10**17, 10**17.5, 10**18, 10**18.5, 10**19, 10**19.5]:
            ax0.axvline(e_cut, c = "gray", ls = "--")

        ax0.set_xlabel("Primary energy / eV")
        ax0.set_xscale("log")

        # Zenith distribution
        ax1.set_title("Zenith distribution")
        ax1.hist(all_zenith, histtype = "step", bins = np.linspace(0, 90, 100), label = "All showers")
        ax1.hist(sel_zenith, histtype = "step", bins = np.linspace(0, 90, 100), label =" Selected showers")
        ax1.set_xlabel("Zenith / $^\circ$")
        ax1.legend(fontsize = 16)

        # Charge integral
        ax2.set_title("Deposited signal in tank")
        n, _, _ = ax2.hist(all_integral, histtype = "step", bins = np.geomspace(1e-1, 1e3, 100), label = "All showers")
        ax2.hist(sel_integral, histtype = "step", bins = np.geomspace(1e-1, 1e3, 100), label = "Selected showers")
        self.ignore_low_VEM and ax2.axvline(self.ignore_low_VEM, ls = "--", c = "gray")
        ax2.set_xlabel("Integral signal / $\mathrm{{VEM}}_\mathrm{{Ch.}}$")
        ax2.set_xscale("log")
        ax2.set_ylim(0, 1.1 * max(n))
        ax2.legend(fontsize = 16)

        # SPDistance distribution
        ax3.set_title("Shower plane distance distribution")
        ax3.hist(all_spd, histtype = "step", bins = np.linspace(0, 6e3, 100), label = "All showers")
        ax3.hist(sel_spd, histtype = "step", bins = np.linspace(0, 6e3, 100), label = "Selected showers")
        ax3.set_xlabel("Shower plane distance / m")
        ax3.legend(fontsize = 16)
        
        # Particle distribution
        ax4.set_title("Particle distribution")
        particle_bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        particle_bins += list(np.geomspace(10, 1e4, 40))
        self.ignore_particles and ax4.axvline(self.ignore_particles + 1, ls = "--", color = "gray")
        ax4.hist(all_muons, histtype = "step", bins = particle_bins, label = "Muons", color = "steelblue")
        ax4.hist(all_electrons, histtype = "step", bins = particle_bins, label = "Electrons", color = "orange")
        ax4.hist(all_photons, histtype = "step", bins = particle_bins, label = "Photons", color = "green")
        ax4.hist(sel_muons, histtype = "step", bins = particle_bins, color = "steelblue", ls = "--")
        ax4.hist(sel_electrons, histtype = "step", bins = particle_bins, color = "orange", ls = "--")
        ax4.hist(sel_photons, histtype = "step", bins = particle_bins, color = "green", ls = "--")
        ax4.plot([],[], c = "k", ls = "--", label = "selected")   
        ax4.set_xlabel("\# of particles")
        ax4.set_xscale("log")
        ax4.set_yscale("log")
        ax4.legend(fontsize = 16)

        # Component information
        ax5.set_title("Component information")
        ax5.set_yscale("log")
        ax5.plot([],[], ls ="--", c = "k", label = "selected")

        x_sig = np.clip(x_sig, -1, 5)
        x_bkg = np.clip(x_bkg, -1, 5)
        ax5.hist(x_sig, bins = 100, histtype = "step", label = "Signal", color = "steelblue")
        n, _, _ = ax5.hist(x_bkg, bins = 100, histtype = "step", label = "Baseline", color = "orange")
        ax5.hist(sel_x_sig, bins = 100, histtype = "step", ls = "--", color = "steelblue")
        ax5.hist(sel_x_bkg, bins = 100, histtype = "step", ls = "--", color = "orange")
        ax5.legend(fontsize = 16)

        ax5.set_ylim(1e1, 1.1 * max(n))
        ax5.set_xlim(-0.2, 5)
        ax5.set_xlabel("Signal strength / VEM")

        plt.subplots_adjust(
            top=0.93,
            bottom=0.111,
            left=0.053,
            right=0.982,
            hspace=0.441,
            wspace=0.302)
        
        self.for_training = temp
        
        if save_dir is not None:
            plt.savefig(save_dir)
        else:
            plt.show()

    # run some dignostics for training purposes
    def training_test(self, n_showers : int = None) -> None :

        if n_showers is None: n_showers = self.__len__()
        temp, self.for_training = self.for_training, True
        SIG, BKG, INT, PAR = 0, 0, 0, 0
        start_time = perf_counter_ns()

        bins = np.linspace(-1, 5, 1000)
        # sig_no_label = np.zeros_like(bins[:-1])
        # sig_label = np.zeros_like(bins[:-1])
        # bkg_hist = np.zeros_like(bins[:-1])
        prior_hist = []
        
        for step_counter, (traces, labels) in enumerate(self):

            progress_bar(step_counter, n_showers, start_time)

            n_sig, n_bkg, n_int, n_prt = self.get_train_loop_statistics()

            # sig_traces, sig_labels = traces[:n_sig], labels[:n_sig]
            # bkg_traces = traces[n_sig:]

            # # evaluate signal component
            # for trace, label in zip (sig_traces, sig_labels):
            #     if label[1]: sig_label += np.histogram(trace, bins = bins)[0]
            #     else: sig_no_label += np.histogram(trace, bins = bins)[0]

            # # evaluate background component
            # for trace in bkg_traces:
            #     bkg_hist += np.histogram(trace, bins = bins)[0]

            prior_hist.append( (n_sig) / (n_sig + n_bkg + n_int + n_prt) )           
            SIG, BKG, INT, PAR = SIG + n_sig, BKG + n_bkg, INT + n_int, PAR + n_prt

            # first use of an assignment expression for me, neat!
            # if step_counter := step_counter + 1 >= n_traces: break
            if step_counter >= n_showers: break

        # # Plot test statistics
        # plt.title("Sliding window trace characteristics")
        # plt.plot(0.5 * (bins[1:] + bins[:-1]), sig_label, label = "Classified signal")
        # plt.plot(0.5 * (bins[1:] + bins[:-1]), sig_no_label, label = "Classified Background")
        # plt.plot(0.5 * (bins[1:] + bins[:-1]), bkg_hist, label = "True Background")
        # plt.axvline(self.ignore_low_VEM, c = "gray", ls = "--")
        # plt.xlabel("Signal strength / VEM")
        # plt.yscale("log")
        # plt.legend()

        plt.figure()
        plt.title("Distribution of priors")
        plt.hist(prior_hist, histtype = "step", bins = np.linspace(0, 1, 60))
        plt.axvline(self.prior, c = "gray", ls = "--")
        plt.xlim(0, 1), plt.ylim(-0.01)

        # Print test statistics
        print("\n\n-- Sliding window summary --")
        print(f"{SIG = :6} + {BKG = :6} = {SIG + BKG :6} traces raised")
        print(f"{INT = :6} + {PAR = :6} = {INT + PAR :6} traces rejected")
        print(f"\nset: {self.prior:.3f} <=> {(SIG)/(BKG + SIG + INT + PAR):.3f} :returned")
        print(f"Cut eliminates {(INT + PAR)/(BKG + SIG + INT + PAR) * 100:.0f}% of all traces\n")

        self.for_training = temp
        plt.show()

from .Classifier import *