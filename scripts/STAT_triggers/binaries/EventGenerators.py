import tensorflow as tf
import numpy as np
import warnings
import random
import typing
import sys, os

warnings.simplefilter("ignore", UserWarning)                                    # ignore UserWarnings from loading empty files

# custom modules for specific use case
from binaries.Signal import VEMTrace

# Helper class (the actual generator) called by EventGenerator
class Generator(tf.keras.utils.Sequence):

    labels = {"SIGNAL": tf.keras.utils.to_categorical(1, 2, dtype = int),       # Signal label
              "BACKGROUND": tf.keras.utils.to_categorical(0, 2, dtype = int)}   # Background label

    # this shouldn't be initialized manually. Use EventGenerator instead!
    def __init__(self, files : str, args : list):

        r'''
        :files ``list[str]``: Path to the individual event files

        :Positional arguments (in order):
        * *pooling* (``bool``) -- apply max pooling to 3 PMT tuple to reduce data size
        * *prior* (``float``) -- p(signal), p(background) = 1 - prior

        * *ADC_to_VEM* (``float``) -- ADC to VEM conversion factor, for ub <-> uub (i think), for example
        * *n_bins* (``int``) -- generate a baseline with <trace_length> bins
        * *sigma* (``float``) -- baseline std in ADC counts ()
        * *mu* (``list``) -- mean ADC level limit [low, high] in ADC counts
        '''

        self.files = files
        self.pooling, self.prior = args[0], args[1]                             # important for __getitem__
        self.ADC_to_VEM_factor = args[2]                                        # conversion factor from ADC to VEM
        self.trace_length = args[3]                                             # number of bins in the VEM trace
        self.baseline_std = args[4]                                             # standard deviation of the baseline
        self.baseline_mean = args[5]                                            # max/min value of the baseline mean

    # one batch of data == one event (with multiple stations, PMTs)
    def __getitem__(self, index : int, for_training : bool = True) -> tuple :

        grouped_events = []                                                     # [[trace1, label1], [trace2, label2], .... ]
        background_proxies = 2                                                  # number of background traces raised for empty file

        try:
            if self.prior == 0:
                raise ZeroDivisionError                                         # if self.prior = 0 only background traces should be raised
            else:
                event_file = self.files[index]                                  # the even file from which to fetch data from

            # check if the signal file is not empty first and foremost
            signals = np.loadtxt(event_file)                                    # get vem trace data

        
            signals = np.split(signals, len(signals) / 3 )                      # group them by station (since there are 3 PMTs)

            # create and add all signal events to batch first
            for signal in signals:
                Label = self.labels["SIGNAL"]

                try:
                    Trace = VEMTrace(signal, ADC_to_VEM = self.ADC_to_VEM_factor, n_bins = self.trace_length, sigma = self.baseline_std, mu = self.baseline_mean)
                except ValueError:
                    sys.exit(f"{event_file} seems to be corrupted, I better stop what I'm doing")
                    
                grouped_events.append([Trace(self.pooling, for_training), Label])

            # then create all background events afterwards
            background_proxies = int(len(signals) * (1/self.prior - 1))
            raise ZeroDivisionError

        except ZeroDivisionError:                                               # ZeroDivisionError means an empty trace file
            for i in range(background_proxies):                                 # raise background_proxy number of backgrounds
                Label = self.labels["BACKGROUND"]
                Trace = VEMTrace(ADC_to_VEM = self.ADC_to_VEM_factor, n_bins = self.trace_length, sigma = self.baseline_std, mu = self.baseline_mean)
                grouped_events.append([Trace(self.pooling, for_training), Label])

        random.shuffle(grouped_events)                                          # shuffle background and signal events around
        traces, labels = zip(*grouped_events)                                   
        if for_training:                                                        # idk why expand_dims is necessary to be honest
            traces = tf.expand_dims(traces, axis = -1)                          # something with the data shape for model.fit         

        return np.array(traces), np.array(labels)

    # returns the number of batches per epoch
    def __len__(self) -> int :
        return len(self.files)

    # called by model.fit at the end of each epoch
    def on_epoch_end(self) -> typing.NoReturn : 
        random.shuffle(self.files)

# Generator class for NN sequential model with some additional functionalities
# This website helped tremendously with writing a working example: shorturl.at/fFI09
class EventGenerator():

    # defaults for the Generator class; Can be overwritten in __new__
    Pooling = True                                                              # whether or not to (max- ) pool trace data
    Split   =  0.8                                                              # Ratio of the training / validation events
    Seed    = False                                                             # make RNG dice rolls reproducible via seed
    Prior   = 0.5                                                               # Probability of a signal event in the data

    # dict for easier adding of different data libraries
    libraries = {
        # "20_20.2" : "/cr/tempdata01/filip/QGSJET-II/protons/20_20.2/",
        "19.5_20" : "/cr/tempdata01/filip/VEM/QGSJET-II/protons/19.5_20/",
        "19_19.5" : "/cr/tempdata01/filip/VEM/QGSJET-II/protons/19_19.5/",
        "18.5_19" : "/cr/tempdata01/filip/VEM/QGSJET-II/protons/18.5_19/",
        "18_18.5" : "/cr/tempdata01/filip/VEM/QGSJET-II/protons/18_18.5/",
        "17.5_18" : "/cr/tempdata01/filip/VEM/QGSJET-II/protons/17.5_18/",
        "17_17.5" : "/cr/tempdata01/filip/VEM/QGSJET-II/protons/17_17.5/",
        "16.5_17" : "/cr/tempdata01/filip/VEM/QGSJET-II/protons/16.5_17/",
        "16_16.5" : "/cr/tempdata01/filip/VEM/QGSJET-II/protons/16_16.5/"
    }

    def __new__(self, datasets : typing.Union[list, str], **kwargs) -> typing.Union[tuple, Generator] :

        r'''
        :datasets ``list[str]``: number of libraries you want included. "all" includes everything.

        :Keyword arguments:
        * *pooling* (``bool``) -- apply max pooling to 3 PMT tuple to reduce data size
        * *split* (``float``) -- fraction of of training set/entire set if split = 1 or 0, only one set is returned
        * *seed* (``bool``) -- fix randomizer seed for reproducibility later on
        * *prior* (``float``) -- p(signal), p(background) = 1 - prior

        Borrowed from VEM trace class
        * *ADC_to_VEM* (``float``) -- ADC to VEM conversion factor, for ub <-> uub (i think), for example
        * *n_bins* (``int``) -- generate a baseline with <trace_length> bins
        * *sigma* (``float``) -- baseline std in ADC counts ()
        * *mu* (``list``) -- mean ADC level limit [low, high] in ADC counts
        '''

        # Set the Generator defaults first and foremost
        self.Pooling = self.set_generator_attribute(self, kwargs, 'pooling', EventGenerator.Pooling)
        self.Split = self.set_generator_attribute(self, kwargs, 'split', EventGenerator.Split)
        self.Seed = self.set_generator_attribute(self, kwargs, 'seed', EventGenerator.Seed)
        self.Prior = self.set_generator_attribute(self, kwargs, 'prior', EventGenerator.Prior)

        # Set the VEM Trace defaults
        ADC_to_VEM_factor = self.set_generator_attribute(self, kwargs, "ADC_to_VEM", VEMTrace.ADC_to_VEM_factor)
        trace_length = self.set_generator_attribute(self, kwargs, "n_bins", VEMTrace.baseline_length)
        baseline_std = self.set_generator_attribute(self, kwargs, "sigma", VEMTrace.baseline_std)
        baseline_mean = self.set_generator_attribute(self, kwargs, "mu", VEMTrace.baseline_mean)

        # set RNG seed if desired
        if self.Seed:
            random.seed(0)      # does this perhaps already fix the numpy seeds?
            np.random.seed(0)   # numpy docs says this is legacy, maybe revisit?

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

        generator_options = [self.Pooling, self.Prior, ADC_to_VEM_factor, trace_length, baseline_std, baseline_mean]

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
                    raise ValueError
            
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