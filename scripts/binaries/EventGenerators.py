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
        * *split* (``float``) -- fraction of of training set/entire set
        * *fix_seed* (``bool``) -- fix randomizer seed for reproducibility
        * *shuffle* (``bool``) -- shuffle event list at the end of generation
        * *pooling* (``bool``) -- apply max pooling to 3 PMT tuple to reduce data size
        * *prior (``float``) -- p(signal), p(background) = 1 - CLASS_IMBALANCE
        * *trace_length* (``int``) -- number of bins in the trace
        * *baseline_std* (``float``) -- baseline std in VEM counts
        * *baseline_mean* (``list``) -- mean ADC level limit [low, high]
    '''

    def __new__(self, dataset : str, **kwargs) -> typing.Any :

        class Generator(tf.keras.utils.Sequence):

            def __init__(self, dataset, *args):

                # select files for specific event generator
                unique_events = os.listdir(os.environ.get('DATA') + dataset)
                start = 0 if not args[0] else int(args[1] * len(unique_events))
                stop = int(args[1] * len(unique_events)) if not args[0] else -1

                # specify dataset architecture
                self.__files = unique_events[start : stop]
                self.__path_to_dataset_folder = dataset
                self.__shuffle, self.__shape = args[2], args[3]
                self.__std, self.__mean = args[4], args[5]
                self.__prior = args[6]
                self.__pooling = args[7]

            # one batch of data == one event (with multiple stations, PMTs)
            def __getitem__(self, index) -> tuple :

                signal_traces = np.loadtxt(os.environ.get('DATA') + self.__path_to_dataset_folder + self.__files[index])        # get vem trace data
                signal_traces = np.split(signal_traces, len(signal_traces)/3 )                                                  # group them by station
                grouped_events = []

                # 0 = background label
                # 1 = signal label

                # Create all signal events
                for station in signal_traces:
                    Trace = VEMTrace("SIG", self.__shape, self.__std, self.__mean, self.__pooling, station)
                    Label = tf.keras.utils.to_categorical(1, 2, dtype = int)
                    grouped_events.append([Trace(), Label])

                # Calculate amount of BKG events required to get desired class imbalance
                N_bkg = len(grouped_events) / self.__prior - len(grouped_events)

                # Create all background events
                for station in range(int(N_bkg)):
                    Trace = VEMTrace("BKG", self.__shape, self.__std, self.__mean, self.__pooling)
                    Label = tf.keras.utils.to_categorical(0, 2, dtype = int)
                    grouped_events.append([Trace(), Label])

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

        def set_kwarg(kwargs, key, fallback):

            try:
                value = kwargs[key]
            except KeyError:
                value = fallback

            return value

        # set binaries for baseline production
        train     = set_kwarg(kwargs, 'train', True)                            # default to creating training set (Train / Validation)
        split     = 1 if not train else set_kwarg(kwargs, 'split', 0.8)         # default to splitting 80% / 20% (Train / Validation)
        pooling   = set_kwarg(kwargs, 'pooling', True)                          # default to max pooling between 3 PMT bins
        fix_seed  = set_kwarg(kwargs, 'fix_seed', False)                        # default to randomizing baseline every time
        shuffle   = set_kwarg(kwargs, 'shuffle', True)                          # default to shuffling data at the end of an epoch
        imbalance = set_kwarg(kwargs, 'prior', 0.5)                             # default to 50%/50% distribution of signal/baseline 
        bkg_size  = set_kwarg(kwargs, 'trace_length', 20000)                    # default to 20 000 bin (166 us) trace size
        bkg_std   = set_kwarg(kwargs, 'baseline_std', 0.5)                      # default to 0.5 ADC counts baseline std
        bkg_mean  = set_kwarg(kwargs, 'baseline_mean', (-0.5, 0.5))             # default to (-0.5, 0.5) limit for baseline mean

        if fix_seed:
            np.random.seed(0)

        TestingSet = Generator(dataset, train, 1 - split, shuffle, bkg_size, bkg_std, bkg_mean, imbalance, pooling)

        if train:
            ValidationSet = Generator(dataset, train, split, shuffle, bkg_size, bkg_std, bkg_mean, imbalance, pooling)
            return TestingSet, ValidationSet
        elif not train:
            return TestingSet