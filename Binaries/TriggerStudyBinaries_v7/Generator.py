from time import perf_counter_ns
import tensorflow as tf
import typing
import random
import sys

from .__config__ import *
from .Signal import *

# See this website for help on a working example: shorturl.at/fFI09
class EventGenerator():

    labels = \
    {
        0: tf.keras.utils.to_categorical(0, 2, dtype = int),                              # Background
        1: tf.keras.utils.to_categorical(1, 2, dtype = int)                               # Signal
    }

    libraries = \
    {
        "16_16.5" : "/cr/tempdata01/filip/QGSJET-II/protons/16_16.5/",
        "16.5_17" : "/cr/tempdata01/filip/QGSJET-II/protons/16.5_17/",
        "17_17.5" : "/cr/tempdata01/filip/QGSJET-II/protons/17_17.5/",
        "17.5_18" : "/cr/tempdata01/filip/QGSJET-II/protons/17.5_18/",
        "18_18.5" : "/cr/tempdata01/filip/QGSJET-II/protons/18_18.5/",
        "18.5_19" : "/cr/tempdata01/filip/QGSJET-II/protons/18.5_19/",
        "19_19.5" : "/cr/tempdata01/filip/QGSJET-II/protons/19_19.5/"
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
        * *q_peak* (``float``) -- ADC to VEM conversion factor, for UB <-> UUB
        * *q_charge* (``float``) -- Conversion factor for the integral trace
        * *n_bins* (``int``) -- generate a baseline with <trace_length> bins
        * *force_inject* (``int``) -- force the injection of <force_inject> background particles
        * *sigma* (``float``) -- baseline std in ADC counts, ignored for real_background
        * *mu* (``list``) -- mean ADC level in ADC counts, ignored for real_background

        __:Classifier:______________________________________________________________

        * *window* (``int``) -- the length of the sliding window
        * *step* (``int``) -- step size of the sliding window analysis
        * *ignore_low_vem* (``float``) -- intentionally mislabel low signal
        '''

        # set all desired environmental variables
        split = kwargs.get("split", GLOBAL.split)
        seed = kwargs.get("seed", GLOBAL.seed)
        prior = kwargs.get("prior", GLOBAL.prior)

        q_peak = kwargs.get("q_peak", GLOBAL.q_peak)
        q_charge = kwargs.get("q_charge", GLOBAL.q_charge)
        n_bins = kwargs.get("n_bins", GLOBAL.n_bins)
        baseline_std = kwargs.get("sigma", GLOBAL.baseline_std)
        baseline_mean = kwargs.get("mu", GLOBAL.baseline_mean)
        n_injected = kwargs.get("force_inject", GLOBAL.force_inject )
        real_background = kwargs.get("real_background", GLOBAL.real_background)
        downsampling = kwargs.get("apply_downsampling", GLOBAL.downsampling)

        ignore_low_VEM = kwargs.get("ignore_low_vem", GLOBAL.ignore_low_VEM)
        sliding_window_length = kwargs.get("window", GLOBAL.window)
        sliding_window_step = kwargs.get("step", GLOBAL.step)

        trace_options = [q_peak, q_charge, n_bins, baseline_std, baseline_mean, n_injected, downsampling, real_background]
        classifier_options = [ignore_low_VEM, sliding_window_length, sliding_window_step, prior]
        
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

        # slit files into training and testing set (if needed)
        if split in [0,1]:
            return Generator(all_files, trace_options, classifier_options)
        else:
            split_files_at_index = int(split * len(all_files))
            training_files = all_files[0:split_files_at_index]
            validation_files = all_files[split_files_at_index:-1]

            TrainingSet = Generator(training_files, trace_options, classifier_options)
            TestingSet = Generator(validation_files, trace_options, classifier_options)

            return TrainingSet, TestingSet 

class Generator(tf.keras.utils.Sequence):

    def __init__(self, signal_files : list, trace_options : list, classifier_options : list) :

        self.ignore_low_VEM, self.window_length, self.window_step, self.prior = classifier_options
        self.q_peak, self.q_charge, self.length, self.sigma, self.mu, self.n_injected, self.downsampling, self.use_real_background = trace_options
        self.files = signal_files

        if self.use_real_background:
            self.RandomTraceBuffer = RandomTrace()

    # number of batches in generator
    def __len__(self) -> int : 
        return len(self.files)

    # generator method to create data on runtime
    def __getitem__(self, index : int, full_trace : bool = False) -> tuple[np.ndarray] :

        labels, traces = [], []

        # Construct either gaussian or random trace baseline
        if not self.use_real_background: 
            baseline = Baseline(self.mu, self.sigma, self.length)
        else: 
            
            self.q_peak, self.q_charge, baseline = self.RandomTraceBuffer.get()     # load INT baseline trace
            baseline += np.random.uniform(0, 1, size = (3, self.length))            # convert it to FLOAT now

        # try to raise a valid trace (i.e. with signal)...
        try:
            if self.prior == 0: raise EmptyFileError
            else: event_file = self.files[index]

            for station in SignalBatch(event_file):
                
                # Add together baseline + signal and inject accidental muons
                VEMTrace = Trace([self.q_peak, self.q_charge, self.length, self.sigma, self.mu, self.n_injected, self.downsampling], baseline, station)
                
                if full_trace:
                    traces.append(VEMTrace)
                else:
                    for index in self.__sliding_window__(VEMTrace):
                        pmt_data, n_sig, integral = VEMTrace.get_trace_window((index, index + self.window_length))

                        # mislabel low energy 
                        if self.ignore_low_VEM: n_sig = 0 if integral < self.ignore_low_VEM else n_sig

                        traces.append(pmt_data), labels.append(EventGenerator.labels[1 if n_sig else 0])

        # ... raise a mock background trace if this fails for various reasons
        except EmptyFileError:
            
            VEMTrace = Trace([self.q_peak, self.q_charge, self.length, self.sigma, self.mu, self.n_injected, self.downsampling], baseline)

            if full_trace:
                traces.append(VEMTrace)
            else:
                for index in self.__sliding_window__(VEMTrace, override_prior = True):
                    pmt_data, n_sig, _ = VEMTrace.get_trace_window((index, index + self.window_length))
                    traces.append(pmt_data), labels.append(EventGenerator.labels[1 if n_sig else 0])

        print()
        return np.array(traces), np.array(labels)

    # calculate a sliding window range conforming (in most cases at least) to a given prior
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

    # check properties of the sliding window, prior, start/stop etc.
    def check_sliding_window(self, batch : int) -> None :

        print(f"Sliding window for first trace in {self.files[batch]}:")
        print("-------------------------")

        traces, _ = self.__getitem__(batch, full_trace = True)

        n_sigs, n_bkgs = 0, 0

        for i in self.__sliding_window__(traces[0]):

            window, n_sig, _ = traces[0].get_trace_window((i, i + self.window_length))

            if n_sig: n_sigs += 1
            else: n_bkgs += 1

            print(i, window.shape, n_sig, "->", n_sigs, n_bkgs)

        print(f"\nGiven prior: {self.prior} <-> {n_sigs / (n_sigs + n_bkgs)} this prior")

    # check if __getitem__ returns sensible batches
    def check_output(self, batch : int) -> None :

        traces, _ = self.__getitem__(batch)

        if traces.shape[0] == 0:

            print(f"Something went wrong with batch {batch}: full shape {traces.shape} ")

            raise StopIteration      

    # run some diagnostics to make sure dataset is in order
    def unit_test(self, n_traces : int = None, full_traces : bool = True) -> None :

        start = perf_counter_ns()

        background_hist, signal_hist, baseline_hist, priors = [], [], [], []
        n_signals, n_backgrounds, n_injected = 0, 0, 0
        has_label, has_no_label = [], []

        if n_traces is None: n_traces = self.__len__()
        
        if full_traces:
            for batch in range(n_traces):

                elapsed = perf_counter_ns() - start
                mean_per_step_ms = elapsed / (batch + 1) * 1e-6

                traces, _ = self.__getitem__(batch, full_trace = full_traces)

                print(f"{100 * (batch/n_traces):.2f}% - {mean_per_step_ms:.2f}ms/batch, ETA = {(n_traces - batch) * mean_per_step_ms * 1e-3:.0f}s {traces[0]}", end ="\r")
                
                for trace in traces:

                    if trace.has_accidentals: 
                        background_hist.append(np.mean(trace.Injected))
                        n_injected += len(trace.injections_start)
                    
                    if trace.has_signal: 

                        try:
                            signal_length = trace.signal_end - trace.signal_start
                            sliding_window = self.__sliding_window__(trace)
                            window_length = (sliding_window[-1] - sliding_window[0]) - 2 * self.window_length
                            priors.append(signal_length / window_length)
                        except ZeroDivisionError: priors.append(1)

                        
                        signal_hist.append(np.mean(trace.Signal))
                        n_signals += 1

                    else: n_backgrounds += 1

                    baseline_hist.append(np.mean(trace.Baseline))

                    for index in self.__sliding_window__(trace):

                        i, f = index, index + self.window_length
                        _, n_sig, integral = trace.get_trace_window((i, f))
                        if self.ignore_low_VEM: n_sig = 0 if integral < self.ignore_low_VEM else n_sig

                        if n_sig: has_label.append(integral)
                        else: has_no_label.append(integral)

        histogram_ranges = [(0.01,3), (0.01,2e5), None]
        histogram_titles = ["Injected Background", "Signal", "Baseline"]
        for j, histogram in enumerate([background_hist, signal_hist, baseline_hist]):

            plt.figure()
            plt.title(histogram_titles[j])
            plt.hist(histogram, histtype = "step", range = histogram_ranges[j], bins = 100, lw = 2)
            plt.yscale("log") if j != 2 else None

            plt.xlabel("Signal / VEM")

        plt.figure()
        plt.title("Distribution of priors")
        plt.axvline(self.prior, c = "gray", ls = "--", lw = "2", label = "required")
        plt.hist(priors, range = (0,1), bins = 50, histtype = "step", label = "returned", lw = 2)

        plt.figure()
        plt.title("Sliding window integral")
        plt.axvline(self.ignore_low_VEM, c = "gray", ls = "--", lw = 2, label = "low VEM cut")
        plt.hist(has_no_label, bins = 50, histtype = "step", label = "label - Background", range = (0,20))
        plt.hist(has_label, bins = 50, histtype = "step", label = "label - Signal", range = (0,20))
        plt.xlabel("Integrated signal / VEM")
        plt.legend()

        print(f"\n\nTotal time: {(perf_counter_ns() - start) * 1e-9 :.2f}s - {n_signals + n_backgrounds} traces")
        print(f"n_signal = {n_signals}, n_background = {n_backgrounds}")
        print(f"n_injected = {n_injected} -> {n_injected / (self.length * (n_signals + n_backgrounds) * GLOBAL.single_bin_duration):.2f} Hz background")
        print("")

        plt.show()
