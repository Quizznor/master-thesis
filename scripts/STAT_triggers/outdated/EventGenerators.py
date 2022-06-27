import typing, sys, os
import tensorflow as tf
import numpy as np
import random

# custom modules for specific use case
from binaries.Signal import VEMTrace


# Generator class for NN sequential model with some additional functionalities
# This website helped tremendously with writing a working example: shorturl.at/fFI09
class EventGenerator():

    r'''
    :dataset ``str``:
        location (relative to $DATA) of dataset

    :Keyword arguments:
        * *train* (``bool``) -- split dataset into training and validation set
        * *split* (``float``) -- fraction of of training set/entire set, 0 < split < 1
        * *fix_seed* (``bool``) -- fix randomizer seed for reproducibility
        * *shuffle* (``bool``) -- shuffle event list at the end of generation
        * *pooling* (``bool``) -- apply max pooling to 3 PMT tuple to reduce data size
        * *prior (``float``) -- p(signal), p(background) = 1 - CLASS_IMBALANCE
        * *trace_length* (``int``) -- number of bins in the trace
        * *baseline_std* (``float``) -- baseline std in VEM counts
        * *baseline_mean* (``list``) -- mean ADC level limit [low, high]
    '''

    # HTCONDOR virtualenv has a problem with type declaration
    # def __init__(self, datasets : list[str], **kwargs) -> typing.NoReturn :
    def __init__(self, datasets, **kwargs) -> typing.NoReturn :

        def set_kwarg(kwargs, key, fallback):

            try:
                value = kwargs[key]
            except KeyError:
                value = fallback

            return value

        libraries = {
            "19_19.5" : "/cr/tempdata01/filip/QGSJET-II/protons/19_19.5/",
            "18.5_19" : "/cr/tempdata01/filip/QGSJET-II/protons/18.5_19/",
            "18_18.5" : "/cr/tempdata01/filip/QGSJET-II/protons/18_18.5/",
            "17.5_18" : "/cr/tempdata01/filip/QGSJET-II/protons/17.5_18/",
            "17_17.5" : "/cr/tempdata01/filip/QGSJET-II/protons/17_17.5/",
            "16.5_17" : "/cr/tempdata01/filip/QGSJET-II/protons/16.5_17/",
            "16_16.5" : "/cr/tempdata01/filip/QGSJET-II/protons/16_16.5/"
        }

        # get chosen datasets
        chosen_datasets = []

        if isinstance(datasets, str):
            try:
                chosen_datasets.append(libraries[datasets])
            except KeyError:
                chosen_datasets = [*libraries.values()]
        elif isinstance(datasets, list):
            for key in datasets:
                try:
                    chosen_datasets.append(libraries[key])
                except KeyError:
                    print(f"COULDN'T FIND '{key}' IN LIBRARY PATHS")
        else:
            sys.exit("COULDN'T CONSTRUCT VALID DATASET")

        # set binaries for baseline production
        split     = set_kwarg(kwargs, 'split', 0.8)                             # default to splitting 80% / 20% (Train / Validation)
        pooling   = set_kwarg(kwargs, 'pooling', True)                          # default to max pooling between 3 PMT bins
        fix_seed  = set_kwarg(kwargs, 'fix_seed', False)                        # default to randomizing baseline every time
        shuffle   = set_kwarg(kwargs, 'shuffle', True)                          # default to shuffling data at the end of an epoch
        imbalance = set_kwarg(kwargs, 'prior', 0.5)                             # default to 50%/50% distribution of signal/baseline 
        bkg_size  = set_kwarg(kwargs, 'trace_length', 20000)                    # default to 20 000 bin (166 us) trace size
        bkg_std   = set_kwarg(kwargs, 'baseline_std', 0.5)                      # default to 0.5 ADC counts baseline std
        bkg_mean  = set_kwarg(kwargs, 'baseline_mean', [-0.5, 0.5])             # default to (-0.5, 0.5) limit for baseline mean

        # make events reproducible
        if fix_seed:
            np.random.seed(0)

        # convert from ADC to VEM_peak
        ADC_to_VEM_factor = 215
        bkg_std = bkg_std / ADC_to_VEM_factor
        bkg_mean = np.array(bkg_mean) / ADC_to_VEM_factor

        # split signal files into training / validation
        self.training_files, self.validation_files = [], []

        for dataset in chosen_datasets:

            signal_events = os.listdir(dataset)
            signal_events.remove("root_files")
            signal_events = [dataset + file for file in signal_events]

            self.training_files.append(signal_events[0 : int( split * len(signal_events))])
            self.validation_files.append(signal_events[int(split * len(signal_events)):-1])

        self.training_files = np.concatenate(self.training_files)
        self.validation_files = np.concatenate(self.validation_files)

        random.shuffle(self.training_files)
        random.shuffle(self.validation_files)

        self.TrainingSet = Generator(self.training_files, split, shuffle, bkg_size, bkg_std, bkg_mean, imbalance, pooling)
        self.ValidationSet = Generator(self.validation_files, -split, shuffle, bkg_size, bkg_std, bkg_mean, imbalance, pooling)

    def __call__(self) -> tuple :

        return self.TrainingSet, self.ValidationSet

# Helper class (the actual generator) called by EventGenerator
class Generator(tf.keras.utils.Sequence):

    def __init__(self, dataset : str, *args):

        # # TODO: this initialization is a bit of a mess
        # # select files for specific event generator
        # unique_events = os.listdir(dataset)
        # start = 0 if args[0] > 0 else int( args[0] * len)
        # stop =  int( args[0] * len) if args[0] > 0 else -1

        # specify dataset architecture
        # self.files = unique_events[start : stop]
        self.files = dataset
        self.__shuffle, self.__shape = args[1], args[2]
        self.__std, self.__mean = args[3], args[4]
        self.__prior = args[5]
        self.__pooling = args[6]

    # one batch of data == one event (with multiple stations, PMTs)
    def __getitem__(self, index : int, for_training : bool = True) -> tuple :

        print(f"\nFetching files from {self.files[index]}", end = "... ")
        grouped_events = []

        # # check if file is not empty first and foremost
        signal_traces = np.loadtxt(self.files[index])                             # get vem trace data

        if signal_traces.shape == (0,):
            Label = tf.keras.utils.to_categorical(0, 2, dtype = int)
            Trace = VEMTrace("BKG", self.__shape, self.__std, self.__mean, self.__pooling)
            grouped_events.append([Trace(), Label])
            traces, labels = zip(*grouped_events)

            if for_training:
                traces = tf.expand_dims(traces, axis=-1)

            return np.array(traces), np.array(labels)

        signal_traces = np.split(signal_traces, len(signal_traces) / 3 )          # group traces by station

        # 0 = background label
        # 1 = signal label

        # Create all signal events
        for station in signal_traces:
            Trace = VEMTrace("SIG", self.__shape, self.__std, self.__mean, self.__pooling, station)
            Label = tf.keras.utils.to_categorical(1, 2, dtype = int)
            
            if for_training:
                grouped_events.append([Trace(), Label])
            else:
                grouped_events.append([Trace, Label])

        # Calculate amount of BKG events required to get desired class imbalance
        N_bkg = len(grouped_events) / self.__prior - len(grouped_events)

        # Create all background events
        for station in range(int(N_bkg)):
            Trace = VEMTrace("BKG", self.__shape, self.__std, self.__mean, self.__pooling)
            Label = tf.keras.utils.to_categorical(0, 2, dtype = int)

            if for_training:
                grouped_events.append([Trace(), Label])
            else:
                grouped_events.append([Trace, Label])

        # shuffle signal and background events
        np.random.shuffle(grouped_events)
        traces, labels = zip(*grouped_events)

        if for_training:
            traces = tf.expand_dims(traces, axis=-1)

        return (np.array(traces), np.array(labels))

    # returns the number of batches per epoch
    def __len__(self) -> int :
        return len(self.files)

    # called by model.fit at the end of each epoch
    def on_epoch_end(self) -> typing.NoReturn : 
        self.__shuffle and np.random.shuffle(self.files)