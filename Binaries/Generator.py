from time import perf_counter_ns
import tensorflow as tf

from .__config__ import *
from .Signal import *

# Wrapper for the Generator class
class EventGenerator():

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
        
        self.for_training = kwargs.get("for_training")
        self.files = signal_files

        # Trace building options
        self.trace_length = kwargs.get("trace_length", GLOBAL.trace_length)
        self.trace_options = \
        {
            "apply_downsampling" : kwargs.get("apply_downsampling", GLOBAL.downsampling),
            "force_inject"       : kwargs.get("force_inject", GLOBAL.force_inject),
            "window_length"      : kwargs.get("window_length", GLOBAL.window),
            "window_step"        : kwargs.get("window_step", GLOBAL.step),
            "trace_length"       : self.trace_length
        }    

        # Generator options
        self.use_real_background = kwargs.get("real_background", GLOBAL.real_background)
        random_index = kwargs.get("random_index", GLOBAL.random_index)
        station = kwargs.get("station", GLOBAL.station)
        self.prior = kwargs.get("prior", GLOBAL.prior)

        # Classifier options
        self.ignore_low_VEM = kwargs.get("ignore_low_vem", GLOBAL.ignore_low_VEM)
        self.ignore_particles = kwargs.get("ignore_particles", GLOBAL.ignore_particles)

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
                    traces.append(window), labels.append(EventGenerator.labels["SIG"])

                # add n_sig * prior background events
                raise NotImplementedError
                # TODO



        if self.prior != 0:
            # should ONLY be used during training, need to return labelled trace windows 
            # where population of SIG and BKG conform to the prior set in __init__    
            if self.for_training:
                return np.array(traces), np.array(labels)
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



        # labels, traces = [], []

        # # Construct either gaussian or random trace baseline
        # if not self.use_real_background: 
        #     baseline = Baseline(self.mu, self.sigma, self.length)
        # else:
        #     self.q_peak, self.q_charge, baseline = self.RandomTraceBuffer.get()     # load INT baseline trace
        #     # baseline += np.random.uniform(0, 1, size = (3, self.length))            # convert it to FLOAT now
        #     # baseline += np.random.uniform(1, 2, size = (3, self.length))            # for testing purposes
        #     # for k in [0, 1, 2]: baseline[k] += np.random.uniform(1, 2)              # for testing purposes

        #     if not self.keep_scale:
        #         self.trace_options[0] = self.q_peak                                  # adjust q_peak for random traces
        #         self.trace_options[1] = self.q_charge                                # adjust q_charge for random traces

        # # try to raise a valid trace (i.e. with signal)...
        # try:

        #     if self.prior == 0: raise EmptyFileError
        #     else: event_file = self.files[index]

        #     for station in SignalBatch(event_file):
                
        #         # Add together baseline + signal and inject accidental muons
        #         VEMTrace = Trace(self.trace_options, baseline, station)                

        #         if full_trace:
        #             traces.append(VEMTrace)

        #         else:
        #             for index in self.__sliding_window__(VEMTrace):
        #                 pmt_data, n_sig, metadata = VEMTrace.get_trace_window((index, index + self.window_length), skip_metadata)
        #                 metadata_per_batch.append(metadata)

        #                 # mislabel low energy 
        #                 if self.ignore_low_VEM: n_sig = 0 if metadata[0] < self.ignore_low_VEM else n_sig

        #                 # mislabel few particles
        #                 if self.ignore_particles: 
        #                     n_sig = 0 if self.ignore_particles >= (VEMTrace.n_muons + VEMTrace.n_electrons + VEMTrace.n_photons) else n_sig 

        #                 traces.append(pmt_data), labels.append(EventGenerator.labels[1 if n_sig else 0])


        # # ... raise a mock background trace if this fails for various reasons
        # except EmptyFileError:

        #     VEMTrace = Trace(self.trace_options, baseline)

        #     if full_trace:
        #         traces.append(VEMTrace)
        #     else:
        #         for index in self.__sliding_window__(VEMTrace, override_prior = True):
        #             pmt_data, n_sig, metadata = VEMTrace.get_trace_window((index, index + self.window_length), skip_metadata)
        #             traces.append(pmt_data), labels.append(EventGenerator.labels[0])
        #             metadata_per_batch.append(metadata)

        # if skip_metadata:
        #     return np.array(traces), np.array(labels)
        # else:
        #     return np.array(traces), np.array(labels), np.array(metadata_per_batch, dtype = object)


    # calculate a sliding window range conforming (in all cases at least) to a given prior
    def __sliding_window__(self, VEMTrace : Trace, override_prior : bool = False) -> range :

        try:
            if override_prior or not VEMTrace.has_signal: raise SlidingWindowError

            # calculate the size of the sliding window needed for a given prior
            signal_length = VEMTrace.signal_end - VEMTrace.signal_start
            
            # sliding in/out adds additional steps with signal, account for this
            length = int(signal_length/self.prior) + 2 * self.window_length

            # return the whole trace if signal is too large for desired prior
            if length + self.window_length > VEMTrace.length: raise SlidingWindowError

            # calculate the exact position of the sliding window range
            # set the sliding window such that the trace is at the end
            stop = min(VEMTrace.signal_end, VEMTrace.length - self.window_length)
            start = stop - length

            # move window to account for over/undershoot of trace window
            if start < 0: start, stop = start - start, stop - start

        # return the whole trace if anything in the previous steps failed
        except SlidingWindowError:
            start, stop = 0, VEMTrace.length - self.window_length

        return range(start, stop, self.window_step)

    # make this class iterable, yields (traces), (labels) iteratively
    def __iter__(self, full_trace : bool = False) -> typing.Generator[tuple, None, StopIteration] : 

        while self.__iteration_index < self.__len__():

            yield self.__getitem__(self.__iteration_index, full_trace, skip_metadata = False)
            self.__iteration_index += 1

        return StopIteration

    # reset the internal state of the generator
    def __reset__(self) -> None : 

        random.shuffle(self.files)
        self.__iteration_index = 0

    # make a copy of this generator (same event files) with different keywords
    def copy(self, **kwargs : dict) -> "Generator" :

        NewGenerator = EventGenerator([], split = 1, **kwargs)
        NewGenerator.files = self.files

        return NewGenerator

    # check properties of the sliding window, prior, start/stop etc.
    def check_sliding_window(self, batch : int) -> None :

        print(f"Sliding window for first trace in {self.files[batch]}:")
        print("-------------------------")

        traces, _ = self.__getitem__(batch, full_trace = True)

        n_sigs, n_bkgs = 0, 0

        for i in self.__sliding_window__(traces[0]):

            window, n_sig, _, _ = traces[0].get_trace_window((i, i + self.window_length), True, True)

            if n_sig: n_sigs += 1
            else: n_bkgs += 1

            print(i, window.shape, n_sig, "->", n_sigs, n_bkgs)

        print(f"\nGiven prior: {self.prior} <-> {n_sigs / (n_sigs + n_bkgs)} this prior")    

    # run some diagnostics to make sure dataset is in order
    def unit_test(self, n_traces : int = None) -> None :

        if n_traces is None: n_traces = self.__len__() * (GLOBAL.n_bins - 10) * 5

        start = perf_counter_ns()
        mean_per_step_us = np.inf
        n_traces_looked_at = 0

        sig_max_hist, bg_max_hist = [], []

        # helper variables for results
        all_energies, all_zeniths = [], []

        # For loop iterates over the whole dataset, return type/format looks like:
        # traces, labels, [ {Integral, SPDistance, Energy, Zenith} , ...]
        try:
            for batch, (traces, labels, metadata) in enumerate(self):

                print(f"{100 * (n_traces_looked_at/n_traces):.2f}% - {mean_per_step_us:.2f}us/trace, ETA = {(n_traces - n_traces_looked_at) * mean_per_step_us * 1e-6:.0f}s", end = 10 * " " + "\r")

                Energy, Zenith = metadata[0, 2:]
                trace_informations = metadata[:, :2]

                all_energies.append(Energy), all_zeniths.append(Zenith)

                for trace, label, event_info in zip(traces, labels, trace_informations):

                    elapsed = perf_counter_ns() - start
                    mean_per_step_us = elapsed / (n_traces_looked_at + 1) * 1e-3

                    n_traces_looked_at += 1

                    # analysis code on trace level
                    if not np.argmax(label):
                        bg_max_hist.append(np.max(trace))
                    else:
                        sig_max_hist.append(np.max(trace))



                    if n_traces_looked_at == n_traces: raise StopIteration
        
        except StopIteration:
            print(f"-- {batch + 1} EVENTS; SUMMARY -------" + 10 * " ")
            print(f"E range: {np.log10(min(all_energies)):.2f} < log(E) < {np.log10(max(all_energies)):.2f}")
            print(f"θ range: {min(all_zeniths):.2f}° <  θ  <  {max(all_zeniths):.2f}°")


        # Energy, Zenith 2d Histogram
        plt.figure()
        plt.xscale("log")
        plt.hist2d(all_energies, all_zeniths, bins = (40, 50))

        # label vs n_particles

        # Trace characteristica
        plt.figure()
        plt.hist(sig_max_hist, color = "orange", histtype = "step", lw = 3, label = "Signal", bins = np.geomspace(0.1, 1e6, 100), density = True)
        plt.hist(bg_max_hist, color = "steelblue", histtype = "step", lw = 3, label = "Background", bins = 100, range = (-1, 3))

        plt.ylim(0, 500)
        plt.legend()
        plt.xscale("log")
        plt.show()


        # background_hist, signal_hist, baseline_hist, priors = [], [], [], []
        # n_signals, n_backgrounds, n_injected, n_p, n_n = 0, 0, 0, 0, 0
        # has_label_integral, has_no_label_integral = [], []
        # has_label_particles, has_no_label_particles = [], []
        # energy_hist, spd_hist = [], []

        # if n_traces is None: n_traces = self.__len__()
        
        # if full_traces:
        #     for batch in range(int(n_traces)):

        #         elapsed = perf_counter_ns() - start
        #         mean_per_step_ms = elapsed / (batch + 1) * 1e-6

        #         traces, _ = self.__getitem__(batch, full_trace = full_traces)

        #         print(f"{100 * (batch/n_traces):.2f}% - {mean_per_step_ms:.2f}ms/batch, ETA = {(n_traces - batch) * mean_per_step_ms * 1e-3:.0f}s {traces[0]}", end ="\r")
                
        #         for trace in traces:

        #             n_particles = 0

        #             if trace.has_accidentals: 
        #                 background_hist.append(np.max(trace.Injected))
        #                 n_injected += len(trace.injections_start)
                    
        #             if trace.has_signal: 

        #                 try:
        #                     n_particles = trace.n_muons + trace.n_electrons + trace.n_photons
        #                     signal_length = trace.signal_end - trace.signal_start
        #                     sliding_window = self.__sliding_window__(trace)
        #                     window_length = (sliding_window[-1] - sliding_window[0]) - 2 * self.window_length
        #                     priors.append(signal_length / window_length)

        #                 except ZeroDivisionError: priors.append(1)

        #                 signal_hist.append(np.max(trace.Signal))
        #                 energy_hist.append(np.log10(trace.Energy))
        #                 spd_hist.append(trace.SPDistance)

        #                 n_signals += 1

        #             else: n_backgrounds += 1

        #             baseline_hist.append(np.mean(trace.Baseline))

        #             for index in self.__sliding_window__(trace):

        #                 i, f = index, index + self.window_length
        #                 _, n_sig, integral, _ = trace.get_trace_window((i, f))

        #                 if self.ignore_low_VEM: n_sig = 0 if integral < self.ignore_low_VEM else n_sig
        #                 if self.ignore_particles: 
        #                     n_sig = 0 if self.ignore_particles >= n_particles else n_sig

        #                 if n_sig: 
        #                     has_label_integral.append(integral)
        #                     has_label_particles.append(n_particles)
        #                     n_p += 1
        #                 else: 
        #                     n_particles = n_particles if self.ignore_particles >= n_particles else 0
        #                     integral = integral if integral < self.ignore_low_VEM else 0
        #                     has_no_label_integral.append(integral)
        #                     has_no_label_particles.append(n_particles)
        #                     n_n += 1

        # histogram_ranges = [(0.01,3), None, None]
        # histogram_titles = ["Injected Background peak", "Signal peak", "Baseline"]
        # for j, histogram in enumerate([background_hist, signal_hist, baseline_hist]):

        #     plt.figure()
        #     plt.title(histogram_titles[j])
        #     plt.hist(histogram, histtype = "step", range = histogram_ranges[j], bins = 100, lw = 2)
        #     plt.yscale("log") if j != 2 else None

        #     plt.xlabel("Signal / VEM")

        # plt.figure("Distribution of energies")        
        # for e in [16.5, 17, 17.5, 18, 18.5, 19]: plt.axvline(e, c = "gray", ls = "--")
        # plt.hist(energy_hist, range = (16, 19.5), bins = 7 * 10, histtype = "step")
        # plt.xlabel("Energy / log( E / eV )")

        # plt.figure("Distribution of shower plane distances")
        # plt.hist(energy_hist, range = (0, 3000), bins = 7 * 10, histtype = "step")
        # plt.xlabel("Shower plane distance")

        # plt.figure()
        # plt.title("Distribution of priors")
        # plt.axvline(self.prior, c = "gray", ls = "--", lw = "2", label = "required")
        # plt.hist(priors, range = (0,1), bins = 50, histtype = "step", label = "returned", lw = 2)

        # plt.figure()
        # plt.title("Sliding window integral")
        # plt.axvline(self.ignore_low_VEM, c = "gray", ls = "--", lw = 2, label = "low VEM cut")
        # plt.hist(has_no_label_integral, bins = 500, histtype = "step", label = f"Background: n = {len(has_no_label_integral)}", range = (-1,20), ls = "--")
        # plt.hist(has_label_integral, bins = 500, histtype = "step", label = f"Signal: n = {len(has_label_integral)}", range = (-1,20), ls = "--")
        # plt.xlabel("Integrated signal / VEM")
        # plt.yscale("log")
        # plt.legend()

        # plt.figure()
        # plt.title("Number of particles")
        # plt.axvline(self.ignore_particles + 1, c = "gray", ls = "--", lw = 2, label = "low particle cut")
        # plt.hist(has_no_label_particles, bins = 21, histtype = "step", label = f"Background: n = {len(has_no_label_particles)}", range = (-1,20), ls = "--")
        # plt.hist(has_label_particles, bins = 21, histtype = "step", label = f"Signal: n = {len(has_label_particles)}", range = (-1,20), ls = "--")
        # plt.xlabel("number of particles")
        # plt.yscale("log")
        # plt.legend()

        # print(f"\n\nTotal time: {(perf_counter_ns() - start) * 1e-9 :.2f}s - {n_signals + n_backgrounds} traces")
        # print(f"n_signal = {n_signals}, n_background = {n_backgrounds}")
        # # print(f"n_classified = {n_p}, n_ignored = {n_n}")
        # print(f"n_injected = {n_injected} -> {n_injected / (self.length * (n_signals + n_backgrounds) * GLOBAL.single_bin_duration):.2f} Hz background")
        # print("")

        # plt.show()
