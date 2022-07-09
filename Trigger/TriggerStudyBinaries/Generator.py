import tensorflow as tf
import numpy as np
import random
import typing
import os

# custom modules for specific use case
from TriggerStudyBinaries.Signal import VEMTrace

# Helper class (the actual generator) called by EventGenerator
# See this websitefor help on a working example: shorturl.at/fFI09
class Generator(tf.keras.utils.Sequence):

    labels = \
    {
        1: tf.keras.utils.to_categorical(1, 2, dtype = int),                    # Signal
        0: tf.keras.utils.to_categorical(0, 2, dtype = int)                     # Background
    }

    # this shouldn't be initialized manually. Use EventGenerator instead!
    def __init__(self, files : str, args : list):

        self.files = files
        self.pooling, self.prior = args[0], args[1]                             # important for __getitem__
        self.window_length, self.window_step = args[2], args[3]                 # specify sliding window parameters
        self.ADC_to_VEM_factor = args[4]                                        # conversion factor from ADC to VEM
        self.trace_length = args[5]                                             # number of bins in the VEM trace
        self.n_injected = args[6]                                               # number of injected stray (e.g.) muons
        self.baseline_std = args[7]                                             # standard deviation of the baseline
        self.baseline_mean = args[8]                                            # max/min value of the baseline mean

        self._signals, self._backgrounds = 0, 0                                 # for bookkeeping purposes

    # one batch of data == one event (with multiple stations, PMTs)
    def __getitem__(self, index : int, reduce : bool = True) -> tuple :

        labels, traces = [], []

        try:
            if self.prior == 0:
                raise ZeroDivisionError                                         # if self.prior = 0 only background traces should be raised
            else:
                event_file = self.files[index]                                  # the event file from which to fetch data from, as string

                if not os.path.getsize(event_file): raise ZeroDivisionError     # check that the file is not empty first and foremost

            with open(event_file, "r") as file:
                t = [[float(x) for x in line.split()] for line in file.readlines()]

            for j, station in enumerate([np.array([t[i],t[i+1],t[i+2]]) for i in range(0,len(t),3)]):

                Trace = VEMTrace(station, n_bins = self.trace_length, sigma = self.baseline_std, mu = self.baseline_mean, force_inject = self.n_injected)

                if reduce:
                    for i in range(*self.get_relevant_trace_window(Trace), self.window_step):
                        label, pmt_data = Trace.get_trace_window(i, self.window_length)
                        labels.append(self.labels[label]), traces.append(pmt_data)

                        print(label, i, i + self.window_length)

                        if label: self._signals += 1                            # for bookkeeping purposes when training a model
                        else: self._backgrounds += 1

                    raise StopIteration
                else:
                    labels.append(self.labels[1]), traces.append(Trace)

        except ZeroDivisionError:

            Trace = VEMTrace(n_bins = self.trace_length, sigma = self.baseline_std, mu = self.baseline_mean, force_inject = self.n_injected)
            start, stop = self.get_relevant_trace_window(Trace)

            if reduce:
                for i in range(0, self.trace_length - self.window_length, self.window_step):
                    label, pmt_data = Trace.get_trace_window(i, self.window_length)
                    labels.append(self.labels[label]), traces.append(pmt_data)
                    self._backgrounds += 1
            else:
                labels.append(self.labels[1]), traces.append(Trace)        

        finally:
            
            print("")
            print("signals",self._signals)
            print("bkg", self._backgrounds)
            return np.array(traces), np.array(labels)

    # returns the number of batches per epoch
    def __len__(self) -> int :
        return len(self.files)

    # return window range that conforms to a chosen prior probability of signal
    def get_relevant_trace_window(self, Trace : VEMTrace) -> tuple :

        signal_length = (Trace._sig_stopped_at - Trace._sig_injected_at)
        n_bkg = int((( self.window_length // self.window_step) +  signal_length // 10) * (1/self.prior - 1))
        start = int(Trace._sig_injected_at - ( n_bkg / 2 * 10 + (self.window_length - 1)))
        stop = int(Trace._sig_stopped_at + ( n_bkg / 2 * 10))
        
        if stop > Trace.trace_length: stop = Trace.trace_length
        if start < 0: start = 0

        print("signal", Trace._sig_injected_at, Trace._sig_stopped_at, Trace._sig_stopped_at - Trace._sig_injected_at)
        print("window",start, stop)
        print(n_bkg)
        print("")


        return start, stop

    # called by model.fit at the end of each epoch
    def on_epoch_end(self) -> typing.NoReturn : 
        random.shuffle(self.files)

        self._signals, self._backgrounds = 0, 0

# Generator class for NN sequential model with some additional functionalities
class EventGenerator():

    # defaults for the Generator class; Can be overwritten in __new__
    Pooling = True                                                              # whether or not to (max- ) pool trace data
    Split   = 0.8                                                               # Ratio of the training / validation events
    Seed    = False                                                             # make RNG dice rolls reproducible via seed
    Prior   = 0.5                                                               # Probability of a signal event in the data
    Window  = 120                                                               # Length (in bins) of the sliding window
    Step    = 10                                                                # Sliding window analysis step size (in bins)

    # dict for easier adding of different data libraries
    libraries = \
    {
        "19_19.5" : "/cr/tempdata01/filip/QGSJET-II/protons/19_19.5/",
        "18.5_19" : "/cr/tempdata01/filip/QGSJET-II/protons/18.5_19/",
        "18_18.5" : "/cr/tempdata01/filip/QGSJET-II/protons/18_18.5/",
        "17.5_18" : "/cr/tempdata01/filip/QGSJET-II/protons/17.5_18/",
        "17_17.5" : "/cr/tempdata01/filip/QGSJET-II/protons/17_17.5/",
        "16.5_17" : "/cr/tempdata01/filip/QGSJET-II/protons/16.5_17/",
        "16_16.5" : "/cr/tempdata01/filip/QGSJET-II/protons/16_16.5/"
    }

    def __new__(self, datasets : typing.Union[list, str], **kwargs) -> typing.Union[tuple, Generator] :

        r'''
        :datasets ``list[str]``: number of libraries you want included. "all" includes everything.

        :Keyword arguments:
        * *pooling* (``bool``) -- apply max pooling to 3 PMT tuple to reduce data size
        * *split* (``float``) -- fraction of of training set/entire set if split = 1 or 0, only one set is returned
        * *seed* (``bool``) -- fix randomizer seed for reproducibility later on
        * *prior* (``float``) -- p(signal), p(background) = 1 - prior
        * *window* (``int``) -- the length of the sliding window = input size of the NN
        * *step* (``int``) -- step size of the sliding window analysis (in bins)

        Borrowed from VEM trace class
        * *ADC_to_VEM* (``float``) -- ADC to VEM conversion factor, for ub <-> uub (i think), for example
        * *n_bins* (``int``) -- generate a baseline with <trace_length> bins
        * *force_inject* (``int``) -- force the injection of <force_inject> pbackground particles
        * *sigma* (``float``) -- baseline std in ADC counts ()
        * *mu* (``list``) -- mean ADC level limit [low, high] in ADC counts
        '''

        # Set the Generator defaults first and foremost
        self.Pooling = self.set_generator_attribute(self, kwargs, 'pooling', EventGenerator.Pooling)
        self.Split = self.set_generator_attribute(self, kwargs, 'split', EventGenerator.Split)
        self.Seed = self.set_generator_attribute(self, kwargs, 'seed', EventGenerator.Seed)
        self.Prior = self.set_generator_attribute(self, kwargs, 'prior', EventGenerator.Prior)
        self.Window = self.set_generator_attribute(self, kwargs, 'window', EventGenerator.Window)
        self.Step = self.set_generator_attribute(self, kwargs, 'step', EventGenerator.Step)

        # Set the VEM Trace defaults
        ADC_to_VEM_factor = self.set_generator_attribute(self, kwargs, "ADC_to_VEM", VEMTrace.ADC_to_VEM_factor)
        trace_length = self.set_generator_attribute(self, kwargs, "n_bins", VEMTrace.trace_length)
        baseline_std = self.set_generator_attribute(self, kwargs, "sigma", VEMTrace.baseline_std)
        baseline_mean = self.set_generator_attribute(self, kwargs, "mu", VEMTrace.baseline_mean)
        n_injected = self.set_generator_attribute(self, kwargs, "force_inject", -1 )

        # set RNG seed if desired
        if self.Seed:
            random.seed(1)      # does this perhaps already fix the numpy seeds?
            np.random.seed(1)   # numpy docs says this is legacy, maybe revisit?

        selected_libraries = self.get_libraries(datasets)
        self.training_files, self.validation_files = [], []

        # add files to both training and validation set
        for library in selected_libraries:

            library_files = [library + file for file in os.listdir(library)]
            
            # to make sure the sample is truly "random"
            library_files.remove(library + "root_files")
        
            random.shuffle(library_files)
            cut_position = int(self.Split * len(library_files))

            # add files to both datasets
            self.training_files.append(library_files[0:cut_position])
            self.validation_files.append(library_files[cut_position:-1])

        # shuffle different libraries with each other
        self.training_files = np.concatenate(self.training_files)
        self.validation_files = np.concatenate(self.validation_files)
        random.shuffle(self.training_files), random.shuffle(self.validation_files)

        generator_options = [self.Pooling, self.Prior, self.Window, self.Step, ADC_to_VEM_factor, trace_length, n_injected, baseline_std, baseline_mean]

        if 0 < self.Split < 1:
            TrainingSet = Generator(self.training_files, generator_options)
            ValidationSet = Generator(self.validation_files, generator_options)
            return TrainingSet, ValidationSet
        else:
            return Generator(np.array(np.concatenate([self.training_files, self.validation_files])), generator_options)

    # helper function for easier handling of kwargs upon initialization
    def set_generator_attribute(self, dict, key, fallback):
        try:
            # check whether kwargs make sense in the first place
            if key == "split" or key == "prior":
                if 0 <= dict[key] <= 1:
                    pass
                else:
                    raise ValueError("Split or Prior must be within [0,1]")
                
            return dict[key]

        except KeyError:
            return fallback

    # return the path of selected libraries from dict keys
    @staticmethod
    def get_libraries(libraries : typing.Union[list, str]) -> list :

        selected_libraries = []

        if isinstance(libraries, str):
            try:
                selected_libraries.append(EventGenerator.libraries[libraries])
            except KeyError:
                libraries != "all" and print(f"Couldn't find {libraries} in library dictionary... adding everything")
                selected_libraries = [*EventGenerator.libraries.values()]
        elif isinstance(libraries, list):
            for key in libraries:
                try:
                    selected_libraries.append(EventGenerator.libraries[key])
                except KeyError:
                    print(f"Couldn't find {key} in library dictionary... continuing without")
                    pass

        return selected_libraries