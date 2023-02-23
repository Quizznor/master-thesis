import tensorflow as tf

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
        if isinstance(datasets, str):
            try: data = EventGenerator.libraries[datasets]
            except KeyError:
                if datasets == "all": data = [*EventGenerator.libraries.values()]
                else: sys.exit("Couldn't construct a valid dataset from inputs")
        elif isinstance(datasets, list):
            try: data = [EventGenerator.libraries[key] for key in datasets]
            except KeyError: sys.exit("Couldn't construct a valid dataset from inputs")


        all_files = [[os.path.abspath(os.path.join(library, p)) for p in os.listdir(library)] for library in data]
        all_files = [item for sublist in all_files for item in sublist if not item.endswith("root_files")]        
        
        random.shuffle(all_files)

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

        * *prior* (``float``) -- p(signal), p(background) = 1 - prior
        * *sliding_window_length* (``int``) -- length of the sliding window
        * *sliding_window_step* (``int``) -- stepsize for the sliding window
        * *real_background* (``bool``) -- use real background from random traces
        * *random_index* (``int``) -- which file to use first in random traces
        * *force_inject* (``int``) -- inject <force_inject> background particles
        * *for_training* (``bool``) -- return labelled batches if *True* 

        __:VEM traces:______________________________________________________________

        * *apply_downsampling* (``bool``) -- make UUB traces resembel UB traces
        * *q_peak* (``float``) -- ADC to VEM conversion factor, for UB <-> UUB
        * *q_charge* (``float``) -- Conversion factor for the integral trace
        * *n_bins* (``int``) -- generate a baseline with <trace_length> bins

        __:Classifier:______________________________________________________________

        * *ignore_low_vem* (``float``) -- intentionally mislabel low VEM_charge signals
        * *ignore_particles* (``int``) -- intentionally mislabel few-particle signals

        '''
        
        self.all_kwargs = kwargs                                                                            # for copy functionality
        self.for_training = kwargs.get("for_training")                                                      # return traces AND labels
        self.files = signal_files                                                                           # all signal files in lib
        self.__iteration_index = 0                                                                          # to support iteration

        # Trace building options
        self.trace_length = kwargs.get("trace_length", GLOBAL.trace_length)
        self.force_inject = kwargs.get("force_inject", GLOBAL.force_inject)
        self.trace_options = \
        {
            "apply_downsampling" : kwargs.get("apply_downsampling", GLOBAL.downsampling),
            "window_length"      : kwargs.get("window_length", GLOBAL.window),
            "window_step"        : kwargs.get("window_step", GLOBAL.step),
            "force_inject"       : self.force_inject,
            "trace_length"       : self.trace_length
        }    

        # Generator options
        self.use_real_background = kwargs.get("real_background", GLOBAL.real_background)                    # use random trace baselines
        random_index = kwargs.get("random_index", GLOBAL.random_index)                                      # start at this random file
        station = kwargs.get("station", GLOBAL.station)                                                     # use this random station
        self.prior = kwargs.get("prior", GLOBAL.prior)                                                      # probability of signal traces

        # Classifier options
        self.ignore_low_VEM = kwargs.get("ignore_low_vem", GLOBAL.ignore_low_VEM)                           # integrated signal cut threshold
        self.ignore_particles = kwargs.get("ignore_particles", GLOBAL.ignore_particles)                     # particle in tank cut threshold

        if self.use_real_background:
            self.RandomTraceBuffer = RandomTrace(station = station, index = random_index)

        if self.use_real_background and self.force_inject is None: self.force_inject = 0

    # number of batches in generator
    def __len__(self) -> int : 
        return len(self.files)

    # generator method to create data on runtime during e.g. training or other analysis
    def __getitem__(self, index : int) -> typing.Union[tuple[np.ndarray], np.ndarray] :

        r'''
        * *for_training = True* -- used for trace diagnostics, full traces that stem from the same shower, returns: *Traces*
        * *for_training = False* -- should ONLY be used during training, returns labelled batches, returns *(Traces, Labels)*
        '''
        
        # used for trace diagnostics, return full traces that stem from the same shower!
        stations = SignalBatch(self.files[index]) if self.prior != 0 else []                                # load this shower file in memory
        self.__integral_n_rejected, self.__particles_n_rejected = 0, 0                                      # keep track of cut statistics
        self.__n_sig, self.__n_bkg = 0, 0                                                                   # keep track of return types
        full_traces, traces, labels = [], [], []                                                            # reserve space for return values

        for station in stations:

            # create the baseline component for this trace
            if self.use_real_background: q_peak, q_charge, baseline = self.RandomTraceBuffer.get()          # load random trace baseline
            else: 
                baseline = Baseline(GLOBAL.baseline_mean, GLOBAL.baseline_std, self.trace_length)           # create mock gauss. baseline
                q_charge = [GLOBAL.q_charge for _ in range(3)]
                q_peak = [GLOBAL.q_peak for _ in range(3)]

            # create injections at this step as well?
            # TODO ...

            self.trace_options["q_charge"] = q_charge
            self.trace_options["q_peak"] = q_peak

            VEMTrace = Trace(baseline, station, self.trace_options)                                         # create the trace
            full_traces.append(VEMTrace)

            if not self.for_training: continue
            else:

                # add signal data to training batch
                for window in VEMTrace:

                    this_label, _ = self.calculate_label(VEMTrace)
                    self.__n_sig = self.__n_sig + 1

                    traces.append(window), labels.append(Generator.labels[this_label])

                # add background events according to prior
                n_rejected = self.__particles_n_rejected + self.__integral_n_rejected
                self.__n_bkg = max(0, int((self.__n_sig - n_rejected) * ( (1 - self.prior) / self.prior ) - n_rejected))

                while len(traces) < self.__n_sig + self.__n_bkg:
                    
                    # build baseline component for this background trace instance
                    if self.use_real_background: q_peak, q_charge, baseline = self.RandomTraceBuffer.get()
                    else: 
                        baseline = Baseline(GLOBAL.baseline_mean, GLOBAL.baseline_std, self.trace_length)
                        q_charge = [GLOBAL.q_charge for _ in range(3)]
                        q_peak = [GLOBAL.q_peak for _ in range(3)]

                    # create injections at this step as well?
                    # TODO ...

                    self.trace_options["q_charge"] = q_charge
                    self.trace_options["q_peak"] = q_peak

                    BackgroundTrace = Trace(baseline, None, self.trace_options)

                    for window in BackgroundTrace:
                        traces.append(window), labels.append(Generator.labels["BKG"])
                        if len(traces) == self.__n_sig + self.__n_bkg: break

        if self.prior != 0:
            # should ONLY be used during training, need to return labelled trace windows 
            # where population of SIG and BKG conform to the prior set in __init__    
            if self.for_training:

                # # shuffle traces / labels
                # p = np.random.permutation(len(traces))
                # traces, labels = np.array(traces)[p], np.array(labels)[p]
                traces, labels = np.array(traces), np.array(labels)

                return traces, labels
            # in all other cases returning the full trace is better for analysis purposes
            else: return full_traces
        
        # Training with prior = 0 is stupid, hence assume self.for_training = False
        # returns 1 Background trace in __getitem__ if prior is set to zero by user
        else:

            if self.use_real_background: q_peak, q_charge, baseline = self.RandomTraceBuffer.get()          # load random trace baseline
            else: 
                baseline = Baseline(GLOBAL.baseline_mean, GLOBAL.baseline_std, self.trace_length)           # create mock gauss. baseline
                q_charge = [GLOBAL.q_charge for _ in range(3)]
                q_peak = [GLOBAL.q_peak for _ in range(3)]

            # create injections at this step as well?
            # TODO ...

            self.trace_options["q_charge"] = q_charge
            self.trace_options["q_peak"] = q_peak

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

        if not VEMTrace.has_signal: return "BKG"
        else:

            integral = VEMTrace.integral_window[VEMTrace._iteration_index]
            if np.isnan(integral): raise ValueError

            # check particles first (computationally easier)
            if self.ignore_particles:
                n_particles = VEMTrace.n_muons + VEMTrace.n_electrons + VEMTrace.n_photons
                if n_particles > self.ignore_particles:

                    self.__particles_n_rejected += 1
                    return "BKG", integral

            # check integral trace next for cut threshold
            if self.ignore_low_VEM:
                if integral > self.ignore_low_VEM: 
                    self.__integral_n_rejected += 1
                    return "BKG", integral

            return "SIG", integral

    # make a copy of this generator (same event files) with different keywords
    def copy(self, **kwargs : dict) -> "Generator" :

        new_kwargs = self.all_kwargs
        for kwarg in kwargs:
            new_kwargs[kwarg] = kwargs[kwarg]

        NewGenerator = Generator(self.files, **new_kwargs)

        return NewGenerator

    # getter for __getitem__ statistics during iteration
    def get_train_loop_statistics(self) -> tuple : 
        return self.__n_sig, self.__n_bkg, self.__integral_n_rejected, self.__particles_n_rejected

    # run some diagnostics on physical variables
    def unit_test(self, n_showers : int = None) -> None :

        if n_showers is None: n_showers = self.__len__()
        temp, self.for_training = self.for_training, False
        n_muons, n_electrons, n_photons = [], [], []
        energy, zenith, spd = [], [], []
        start_time = perf_counter_ns()
        x_sig, x_bkg = [], []

        for batch, Shower in enumerate(self):
            
            progress_bar(batch, n_showers, start_time)
            
            for trace in Shower:

                # Shower metadata
                energy.append(trace.Energy)
                zenith.append(trace.Zenith)
                spd.append(trace.SPDistance)
                n_muons.append(trace.n_muons)
                n_electrons.append(trace.n_electrons)
                n_photons.append(trace.n_photons)

                # Particles...?
                # TODO ...

                # Signal component
                x_sig.append(np.mean(trace.Signal))


                # Baseline component
                x_bkg.append(np.mean(trace.Baseline))

                # Injection component
                # TODO ...

            if batch == n_showers: break

        plt.figure()
        plt.title("Component information")
        plt.yscale("log")

        x_sig = np.clip(x_sig, -1, 5)
        x_bkg = np.clip(x_bkg, -1, 5)
        plt.hist(x_sig, bins = 100, histtype = "step", label = "Signal", density = True)
        plt.hist(x_bkg, bins = 100, histtype = "step", label = "Baseline", density = True)
        plt.legend()

        plt.xlabel("Signal strength / VEM")
        plt.ylabel("Occupation probability")

        _, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)

        # Energy distribution
        ax0.set_title("Energy distribution")
        ax0.hist(energy, histtype = "step", bins = np.geomspace(10**16, 10**19.5, 100))

        for e_cut in [10**16, 10**16.5, 10**17, 10**17.5, 10**18, 10**18.5, 10**19, 10**19.5]:
            ax0.axvline(e_cut, c = "gray", ls = "--")

        ax0.set_xlabel("Primary energy / eV")
        ax0.set_xscale("log")

        # Zenith distribution
        ax1.set_title("Zenith distribution")
        ax1.hist(zenith, histtype = "step", bins = np.linspace(0, 90, 100))
        ax1.set_xlabel("Zenith / °")

        # SPDistance distribution
        ax2.set_title("Shower plane distance distribution")
        ax2.hist(spd, histtype = "step", bins = np.linspace(0, 6e3, 100))
        ax2.set_xlabel("Shower plane distance / m")
        
        # Particle distribution
        ax3.set_title("Particle distribution")
        particle_bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        particle_bins += list(np.geomspace(10, 1e4, 40))
        ax3.hist(n_muons, histtype = "step", bins = particle_bins, label = "Muons")
        ax3.hist(n_electrons, histtype = "step", bins = particle_bins, label = "Electrons")
        ax3.hist(n_photons, histtype = "step", bins = particle_bins, label = "Photons")
        ax3.set_xlabel("Number of particles")
        ax3.set_xscale("log")
        ax3.legend()

        plt.tight_layout()

        self.for_training = temp

    # run some dignostics for training purposes
    def training_test(self, n_showers : int = None) -> None :

        if n_showers is None: n_showers = self.__len__()
        temp, self.for_training = self.for_training, True
        SIG, BKG, INT, PAR = 0, 0, 0, 0
        start_time = perf_counter_ns()

        bins = np.linspace(-1, 5, 1000)
        sig_no_label = np.zeros_like(bins[:-1])
        sig_label = np.zeros_like(bins[:-1])
        bkg_hist = np.zeros_like(bins[:-1])
        prior_hist = []
        
        for step_counter, (traces, labels) in enumerate(self):

            progress_bar(step_counter, n_showers, start_time)

            n_sig, n_bkg, n_integral_rejected, n_particles_rejected = self.get_train_loop_statistics()
            sig_traces, sig_labels = traces[:n_sig], labels[:n_sig]
            bkg_traces = traces[n_sig:]

            # evaluate signal component
            for trace, label in zip (sig_traces, sig_labels):
                if label[1]: sig_label += np.histogram(trace, bins = bins)[0]
                else: sig_no_label += np.histogram(trace, bins = bins)[0]

            # evaluate background component
            for trace in bkg_traces:
                bkg_hist += np.histogram(trace, bins = bins)[0]

            prior_hist.append( (n_sig - n_integral_rejected - n_particles_rejected) / (n_sig + n_bkg) )           
            SIG, BKG, INT, PAR = SIG + n_sig, BKG + n_bkg, INT + n_integral_rejected, PAR + n_particles_rejected

            # # first use of an assignment expression for me, neat!
            # if step_counter := step_counter + 1 >= n_traces: break
            if step_counter >= n_showers: break

        # Plot test statistics
        plt.title("Sliding window trace characteristics")
        plt.plot(0.5 * (bins[1:] + bins[:-1]), sig_label, label = "Classified signal")
        plt.plot(0.5 * (bins[1:] + bins[:-1]), sig_no_label, label = "Classified Background")
        plt.plot(0.5 * (bins[1:] + bins[:-1]), bkg_hist, label = "True Background")
        plt.axvline(self.ignore_low_VEM, c = "gray", ls = "--")
        plt.xlabel("Signal strength / VEM")
        plt.yscale("log")
        plt.legend()

        plt.figure()
        plt.title("Distribution of priors")
        plt.hist(prior_hist, histtype = "step", bins = np.linspace(0, 1, 30))
        plt.axvline(self.prior, c = "gray", ls = "--")
        plt.xlim(0, 1), plt.ylim(-0.01)

        # Print test statistics
        print("\n\n-- Sliding window summary --")
        print(f"{SIG = } + {BKG = } = {SIG + BKG} traces raised")
        print(f"{INT = } + {PAR = } = {INT + PAR} traces rejected")
        print(f"Prior = {(SIG - INT - PAR)/(BKG + SIG) * 100:.0f}%")

        self.for_training = temp
