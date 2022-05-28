import typing, sys, os
import tensorflow as tf
import numpy as np

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

    def __init__(self, libraries : list[str], **kwargs) -> typing.NoReturn :

        def set_kwarg(kwargs, key, fallback):

            try:
                value = kwargs[key]
            except KeyError:
                value = fallback

            return value

        library_paths = {
            "19_19.5" : "/cr/tempdata01/filip/protons/19_19.5/",
            "18.5_19" : "/cr/tempdata01/filip/protons/18.5_19/",
            "18_18.5" : "/cr/tempdata01/filip/protons/18_18.5/",
            "17.5_18" : "/cr/tempdata01/filip/protons/17.5_18/",
            "17_17.5" : "/cr/tempdata01/filip/protons/17_17.5/",
            "16.5_17" : "/cr/tempdata01/filip/protons/16.5_17/",
        }

        # get chosen datasets
        if isinstance(libraries, str):
            try:
                datasets = list(library_paths[libraries])
            except KeyError:
                datasets = [*library_paths.values]
        elif isinstance(libraries, list):
            datasets = []
            for key in libraries:
                try:
                    datasets.append(library_paths[key])
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

        for dataset in datasets:
            signal_events = os.listdir(dataset)
            self.training_files.append(signal_events[0 : int( split * len(signal_events))])
            self.validation_files.append(signal_events[int(split * len(signal_events)):-1])

        self.TestingSet = Generator(self.training_files, split, shuffle, bkg_size, bkg_std, bkg_mean, imbalance, pooling)
        self.ValidationSet = Generator(self.validation_files, -split, shuffle, bkg_size, bkg_std, bkg_mean, imbalance, pooling)

        def __call__(self) -> tuple :

            return self.TestingSet, self.ValidationSet

# Helper class (the actual generator) called by EventGenerator
class Generator(tf.keras.utils.Sequence):

    def __init__(self, dataset : str, *args):

        # select files for specific event generator
        unique_events = os.listdir(dataset)
        start = 0 if args[0] > 0 else int( args[0] * len)
        stop =  int( args[0] * len) if args[0] > 0 else -1

        # specify dataset architecture
        self.__files = unique_events[start : stop]
        self.__path_to_dataset_folder = dataset
        self.__shuffle, self.__shape = args[2], args[3]
        self.__std, self.__mean = args[4], args[5]
        self.__prior = args[6]
        self.__pooling = args[7]

    # one batch of data == one event (with multiple stations, PMTs)
    def __getitem__(self, index : int, for_training : bool = True) -> tuple :

        # does this still work?
        signal_traces = np.loadtxt("/cr/data01/filip/" + self.__path_to_dataset_folder + self.__files[index])           # get vem trace data
        signal_traces = np.split(signal_traces, len(signal_traces) / 3 )                                                # group them by station
        grouped_events = []

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
        traces = tf.expand_dims(traces, axis=-1)

        return (np.array(traces), np.array(labels))

    # returns the number of batches per epoch
    def __len__(self) -> int :
        return len(self.__files)

    # called by model.fit at the end of each epoch
    def on_epoch_end(self) -> typing.NoReturn : 
        self.__shuffle and np.random.shuffle(self.__files)