import sys, os

sys.dont_write_bytecode = True

import tensorflow as tf
import numpy as np
import typing

# baseline shape defaults. 1 VEM_peak = 61.75 ADC !
# see David's mail from 08.02 for info on magic numbers
BASELINE_LENGTH = 20000                     # number of bins in baseline
BASELINE_STD = 0.5 / 61.75                  # baseline std in VEM counts
BASELINE_MEAN = [-0.5/61.75, +0.5/61.75]    # mean ADC level limits [low, high]

# defaults for DataSetGenerator and EventGenerator
# fwiw, these values are picked because they SHOULD make sense
DATASET_SPLIT = 0.8                         # fraction of training set/entire set
DATASET_FIX_SEED = False                    # fix randomizer seed for reproducibility
DATASET_SHUFFLE = True                      # shuffle event list at the end of generation
DATASET_POOLING = True                      # apply max pooling to PMT data beforehand
CLASS_IMBALANCE = 0.5                       # p(signal), p(background) = 1 - CLASS_IMBALANCE

# defaults for neural network, picked more or less randomly
MODEL_ARCHITECTURE = (4000, 100)            # defines # of dense layers plus # of neurons     
MODEL_LOSS = 'categorical_crossentropy'     # option for custom loss function, default for now
MODEL_OPTIMIZER = 'adam'                    # default optimizer, don't know much about it


# Event wrapper for measurements of a SINGLE tank with 3 PMTs
class VEMTrace():

    def __init__(self, label : str, *args, **kwargs) -> typing.NoReturn :

        # full initialization of trace on first call
        try:

            # set trace shape characteristics first
            # defaults are defined in DataSetGenerator
            self.trace_length = args[0]

            # set baseline std (default exactly 0.5 ADC)
            self.baseline_std = args[1]

            # set baseline mean (default in [-0.5, 0.5] ADC)
            self.baseline_mean = np.random.uniform(*args[2])

            # create baseline VEM trace, same mean and std (TODO different std?)
            self.__pmt_1 = np.random.normal(self.baseline_mean, self.baseline_std, self.trace_length)
            self.__pmt_2 = np.random.normal(self.baseline_mean, self.baseline_std, self.trace_length)
            self.__pmt_3 = np.random.normal(self.baseline_mean, self.baseline_std, self.trace_length)

            if label == "signal": 

                # don't catch exception here, since we _need_ signal data to continue
                signal_length = len(args[3][0])
                vem_signals = args[3]

                assert len(vem_signals[0]) == len(vem_signals[1]) == len(vem_signals[2]), "SIGNAL SHAPES DONT MATCH!\n"
                assert self.trace_length > signal_length, "SIGNAL DOES NOT FIT INTO BASELINE!\n"

                # overlay signal shape at random position of baseline
                start = np.random.randint(-self.trace_length, -signal_length)
                self.__pmt_1[start : start + signal_length] += vem_signals[0]
                self.__pmt_2[start : start + signal_length] += vem_signals[1]
                self.__pmt_3[start : start + signal_length] += vem_signals[2]

            else:
                pass

        # dummy initialization from preprocessed VEMTrace
        except IndexError:

            self.__pmt_1, self.__pmt_2, self.__pmt_3 = np.split(kwargs['trace'], 3)
            self.trace_length = len(self.__pmt_1)
            self.triggered = self.has_triggered()                                       # whether "legacy" triggers would activate

        try:
            self.pooling = kwargs['pooling']                                            # apply manual (max) prepooling if desired
        except KeyError:
            self.pooling = False

        self.label = label                                                              # whether trace is signal or background

    # getter for easier handling of data classes
    def __call__(self) -> tuple :

        if self.pooling:
            return [max([self.__pmt_1[i], self.__pmt_2[i], self.__pmt_3[i]]) for i in range(len(self.__pmt_1))]
        elif not self.pooling:
            return list(self.__pmt_1) + list(self.__pmt_2) + list(self.__pmt_3)

    # Whether or not any of the existing triggers caught this event
    def has_triggered(self) -> bool : 

        # check T1 first, then ToT, for performance reasons
        # have T2 only when T1 also triggered, so ignore it
        T1_is_active = self.absolute_threshold_trigger(1.75)

        if not T1_is_active:
            ToT_is_active = self.time_over_threshold_trigger()

        return T1_is_active or ToT_is_active

    # method to check for (coincident) absolute signal threshold
    def absolute_threshold_trigger(self, threshold : float) -> bool : 

        # hierarchy doesn't (shouldn't?) matter, since we need coincident signal anyway
        for i in range(self.trace_length):
            if self.__pmt_1[i] >= threshold:
                if self.__pmt_2[i] >= threshold:
                    if self.__pmt_3[i] >= threshold:
                        return True
                    else: continue
                else: continue
            else: continue
        
        return False

    # method to check for elevated baseline threshold trigger
    def time_over_threshold_trigger(self) -> bool : 

        window_length = 120      # amount of bins that are being checked
        threshold     = 0.2      # bins above this threshold are 'active'
        
        # count initial active bins
        pmt1_active = len(self.__pmt_1[:window_length][self.__pmt_1[:window_length] > threshold])
        pmt2_active = len(self.__pmt_2[:window_length][self.__pmt_2[:window_length] > threshold])
        pmt3_active = len(self.__pmt_3[:window_length][self.__pmt_3[:window_length] > threshold])

        for i in range(window_length, self.trace_length):

            # check if ToT conditions are met
            ToT_trigger = [pmt1_active >= 13, pmt2_active >= 13, pmt3_active >= 13]

            if ToT_trigger.count(True) >= 2:
                return True

            # overwrite oldest bin and reevaluate
            pmt1_active += self.update_bin_count(i, self.__pmt_1, window_length, threshold)
            pmt2_active += self.update_bin_count(i, self.__pmt_2, window_length, threshold)
            pmt3_active += self.update_bin_count(i, self.__pmt_3, window_length, threshold)

        return False

    @staticmethod
    # helper method for time_over_threshold_trigger
    def update_bin_count(index : int, array: np.ndarray, window_length : int, threshold : float) -> int : 

        # is new bin active?
        if array[index] >= threshold:
            updated_bin_count = 1
        else:
            updated_bin_count = 0

        # was old bin active?
        if array[index - window_length] >= threshold:
            updated_bin_count -= 1

        return updated_bin_count

# Wrapper for easier handling of EventGenerator classes
class DataSetGenerator():
    
    r'''
    :dataset:
        location (relative to $DATA) of dataset
    :type dataset: ``str``

    :Keyword arguments:
        * *train* (``bool``) -- split dataset into training and validation set
        * *split* (``float``) -- fraction of of training set/entire set
        * *fix_seed* (``bool``) -- fix randomizer seed for reproducibility
        * *shuffle* (``bool``) -- shuffle event list at the end of generation
        * *pooling* (``bool``) -- apply max pooling to 3 PMT tuple to reduce data size
        * *class imbalance (``float``) -- p(signal), p(background) = 1 - CLASS_IMBALANCE
        * *trace_length* (``int``) -- number of bins in the trace
        * *baseline_std* (``float``) -- baseline std in VEM counts
        * *baseline_mean* (``list``) -- mean ADC level limit [low, high]
    '''

    def __new__(self, dataset : str, **kwargs) -> tuple :

        def set_kwarg(key : str, fallback : typing.Any, input_validation : typing.Callable = lambda x: True ) -> typing.NoReturn : 

            try: value = kwargs[key]
            except KeyError: value = fallback

            input_validation(value)

            return value

        # set split between training and validation (default 0.8)
        split = set_kwarg('split', DATASET_SPLIT)

        # fix RNG seed if desired (default is desired)
        fix_seed = set_kwarg('fix_seed', DATASET_FIX_SEED)
        fix_seed and np.random.seed(0)

        # shuffle datasets if desired (default is desired)
        shuffle = set_kwarg('shuffle', DATASET_SHUFFLE)

        # pool input data if desired (defaults to pooling)
        pool = set_kwarg('pooling', DATASET_POOLING)

        # set class imbalance (default no imbalance, 50/50 )
        class_imbalance = set_kwarg('class_imbalance', CLASS_IMBALANCE)

        # set baseline length (default 166 μs = 20000 bins)
        input_shape = set_kwarg('trace_length', BASELINE_LENGTH)

        # set baseline std (default exactly 0.5 ADC)
        baseline_std = set_kwarg('baseline_std', BASELINE_STD)

        # set baseline mean (default in [-0.5, 0.5] ADC)
        baseline_mean_limit = set_kwarg('baseline_mean', BASELINE_MEAN)

        try:
            if kwargs['train']:
                raise KeyError
            else:
                Dataset = EventGenerator(dataset, True, 1, shuffle, input_shape, baseline_std, baseline_mean_limit, class_imbalance)

            return Dataset
                
        except KeyError:
            TrainingSet = EventGenerator(dataset, True, split, shuffle, input_shape, baseline_std, baseline_mean_limit, class_imbalance, pooling = pool)
            ValidationSet = EventGenerator(dataset, False, split, shuffle, input_shape, baseline_std, baseline_mean_limit, class_imbalance, pooling = pool)
            
            return TrainingSet, ValidationSet 

# Generator class for NN sequential model with some additional functionalities
# This website helped tremendously with writing a working example: shorturl.at/fFI09
class EventGenerator(tf.keras.utils.Sequence):

    def __init__(self, dataset, *args, **kwargs) -> typing.NoReturn :

        # select files for specific event generator
        unique_events = os.listdir(os.environ.get('DATA') + dataset)
        start = 0 if args[0] else int(args[1] * len(unique_events))
        stop = int(args[1] * len(unique_events)) if args[0] else -1

        # specify dataset architecture
        self.__files = unique_events[start : stop]
        self.__path_to_dataset_folder = dataset
        self.__shuffle, self.__shape = args[2], args[3]
        self.__std, self.__mean = args[4], args[5]
        self.__prior = args[6]
        self.__pooling = kwargs['pooling']

    # returns one batch of data
    # one batch of data == one event (with multiple stations, PMTs)
    # TODO: maybe overthink this to better implement class imbalance?
    def __getitem__(self, index) -> tuple :

        signal_traces = np.loadtxt(os.environ.get('DATA') + self.__path_to_dataset_folder + self.__files[index])
        signal_traces = np.split(signal_traces, len(signal_traces)/3 )
        traces, labels = [], []

        for station in signal_traces:
            Trace = VEMTrace("signal", self.__shape, self.__std, self.__mean, station, pooling = self.__pooling)

            # 0 = background label
            # 1 = signal label

             # decide whether baseline gets interlaced
            choice = np.random.choice([1, 0], p = [self.__prior, 1 - self.__prior])

            while not choice:

                # add background event to list
                BackgroundTrace = VEMTrace("background", self.__shape, self.__std, self.__mean, pooling = self.__pooling)
                labels.append(tf.keras.utils.to_categorical(choice, 2, dtype = int))
                traces.append(BackgroundTrace())

                # add another background event to list
                choice = np.random.choice([1, 0], p = [self.__prior, 1 - self.__prior])
            
            labels.append(tf.keras.utils.to_categorical(choice, 2, dtype = int))
            traces.append(Trace())

        # to include channel size
        traces = tf.expand_dims(traces, axis=-1)

        return (np.array(traces), np.array(labels))

    # returns the number of batches per epoch
    def __len__(self) -> int :
        return len(self.__files)

     # called by model.fit at the end of each epoch
    def on_epoch_end(self) -> typing.NoReturn : 
        self.__shuffle and np.random.shuffle(self.__files)

# Wrapper for tf.keras.Sequential model with some additional functionalities
class Classifier():

    def __init__(self, init_from_disk : str = None) -> typing.NoReturn:

        tf.config.run_functions_eagerly(True)

        if init_from_disk is None:

            self.__epochs = 0
            self.model = tf.keras.models.Sequential()

            # input layer to specify input size
            self.model.add(tf.keras.layers.InputLayer(input_shape=(BASELINE_LENGTH, 1), batch_size=None))

            # architecture of this NN is - apart from in/output - completely arbitrary, at least for now
            self.model.add(tf.keras.layers.Conv1D(filters = 1, kernel_size = 200, activation = 'relu'))
            self.model.add(tf.keras.layers.Conv1D(filters = 1, kernel_size = 20, activation = 'relu'))
            self.model.add(tf.keras.layers.Conv1D(filters = 1, kernel_size = 2, activation = 'softmax'))
        
        elif init_from_disk is not None:

            self.__epochs = int(init_from_disk[init_from_disk.rfind('_') + 1:])                                 # set previously run epochs as start
            self.model = tf.keras.models.load_model(init_from_disk)                                             # load model, doesn't work with h5py 3.x!

        self.model.compile(loss=MODEL_LOSS, optimizer=MODEL_OPTIMIZER, metrics=['accuracy'], run_eagerly=True)  # compile the model and print a summary
        print(self.model.summary())

    # Train the model network on the provided training/validation set
    def train(self, training_set : EventGenerator, validation_set : EventGenerator, epochs : int) -> typing.NoReturn:
        self.model.fit(training_set, validation_data=validation_set, initial_epoch = self.__epochs, epochs = epochs, verbose = 1)
        self.__epochs = epochs

    # Save the model to disk
    def save(self, directory_path : str) -> typing.NoReturn : 
        self.model.save(directory_path + f"model_{self.__epochs}")

    # Predict a batch or single trace
    def predict(self, **kwargs) -> bool :

        # Non-functional for now
        r'''
            :Keyword arguments:
            * *trace* (``tuple``) -- predict type of a single trace
            * *dataset* (``EventGenerator``) -- classify entire dataset
            * *compare_legacy* (``bool``) -- compare NN to threshold/ToT trigger
        '''

        # is this correct? =D
        prediction = self.model.__call__(np.reshape(kwargs['trace'], (1, len(kwargs['trace'])))).numpy()[0]
        return np.argmax(prediction) == 1

    # Wrapper for pretty printing
    def __str__(self) -> str :
        self.model.summary()
        return ""