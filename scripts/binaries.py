import tensorflow as tf
import numpy as np
import typing, os

# baseline shape defaults. 1 VEM_peak = 61.75 ADC !
# see David's mail from 08.02 for info on magic numbers

BASELINE_LENGTH = 20000                 # number of bins in baseline
BASELINE_STD = 0.5 / 61.75              # baseline std in VEM counts
BASELINE_MEAN = [-8.097e-3, +8.097e-3]  # mean ADC level limits [low, high]

# defaults for DataSetGenerator and EventGenerator
# fwiw, these values are picked because they SHOULD™ make sense

DATASET_SPLIT = 0.8                     # fraction of training set/entire set
DATASET_FIX_SEED = True                 # fix randomizer seed for reproducibility
DATASET_SHUFFLE = True                  # shuffle event list at the end of generation


# Event wrapper for measurements of a SINGLE tank with 3 PMTs
class VEMTrace():

    # TODO change kwargs to args maybe
    r'''
    :Keyword arguments:
        * *trace_length* (``int``) -- number of bins in the trace
        * *baseline_std* (``float``) -- baseline std in VEM counts
        * *baseline_mean* (``list``) -- mean ADC level limit [low, high]
        * *signal* (``str``) -- location (relative to $DATA) of signal file
        * *dataset* (``str``) -- location (relative to $DATA) of dataset
    '''

    def __init__(self, label : str, **kwargs) -> None:

        # set baseline length (default 166 μs = 20000 bins)
        try: self.trace_length = kwargs['trace_length']
        except KeyError: self.trace_length = BASELINE_LENGTH

        # set baseline std (default exactly 0.5 ADC)
        try: self.baseline_std = kwargs['baseline_std']
        except KeyError: self.baseline_std = BASELINE_STD

        # set baseline mean (default in [-0.5, 0.5] ADC)
        try: self.baseline_mean = kwargs['baseline_mean']
        except KeyError: self.baseline_mean = np.random.uniform(*BASELINE_MEAN)

        # create baseline VEM trace, same mean and std (TODO different std?)
        self.__pmt_1 = np.random.normal(self.baseline_mean, self.baseline_std, self.trace_length)
        self.__pmt_2 = np.random.normal(self.baseline_mean, self.baseline_std, self.trace_length)
        self.__pmt_3 = np.random.normal(self.baseline_mean, self.baseline_std, self.trace_length)

        if label == "signal": 

            # don't catch exception here, since we _need_ signal data to continue
            vem_signals = np.loadtxt(os.environ.get('DATA') + kwargs['dataset'] + kwargs['signal'])
            signal_length = len(vem_signals[0])

            assert len(vem_signals[0]) == len(vem_signals[1]) == len(vem_signals[2]), "SIGNAL SHAPES DONT MATCH!\n"
            assert self.trace_length > signal_length, "SIGNAL DOES NOT FIT INTO BASELINE!\n"

            # overlay signal shape at random position of baseline
            start = np.random.randint(-self.trace_length, -signal_length)
            self.__pmt_1[start : start + signal_length] += vem_signals[0]
            self.__pmt_2[start : start + signal_length] += vem_signals[1]
            self.__pmt_3[start : start + signal_length] += vem_signals[2]

        else:
            pass

        self.triggered = self.has_triggered()   # whether "legacy" triggers would activate
        self.label = label                      # whether trace is signal or background

    # getter for easier handling of data classes
    def __call__(self) -> tuple :

        return (self.__pmt_1, self.__pmt_2, self.__pmt_3)

    # Whether or not any of the existing triggers caught this event
    def has_triggered(self) -> bool : 

        # check T1 first, then ToT, for performance reasons
        # have T2 only when T1 also triggered, so ignore it
        T1_is_active = self.absolute_threshold_trigger(1.75)

        if not T1_is_active:
            ToT_is_active = self.time_over_threshold_trigger()
        else:
            ToT_is_active = False

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
            pmt1_active += self.updated_bin_count(i, self.__pmt_1, window_length, threshold)
            pmt2_active += self.updated_bin_count(i, self.__pmt_2)
            pmt3_active += self.updated_bin_count(i, self.__pmt_3)

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
        * *split* (``float``) -- fraction of of training set/entire set
        * *fix_seed* (``bool``) -- fix randomizer seed for reproducibility
        * *shuffle* (``bool``) -- shuffle event list at the end of generation
        * *trace_length* (``int``) -- number of bins in the trace
        * *baseline_std* (``float``) -- baseline std in VEM counts
        * *baseline_mean* (``list``) -- mean ADC level limit [low, high]
    '''

    def __new__(self, dataset : str, **kwargs) -> tuple :

        # set split between training and validation (default 0.8)
        try: split = kwargs['split']
        except KeyError: split = DATASET_SPLIT
        assert 0 < split < 1, "PLEASE PROVIDE A VALID SPLIT: 0 < split < 1"

        # fix RNG seed if desired (default is desired)
        try: fix_seed = kwargs['fix_seed']
        except KeyError: fix_seed = DATASET_FIX_SEED
        fix_seed and np.random.seed(0)

        # shuffle datasets if desired (default is desired)
        try: shuffle = kwargs['shuffle']
        except KeyError: shuffle = DATASET_SHUFFLE

        # set baseline length (default 166 μs = 20000 bins)
        try: input_shape = kwargs['trace_length']
        except KeyError: input_shape = BASELINE_LENGTH

        # set baseline std (default exactly 0.5 ADC)
        try: baseline_std = kwargs['baseline_std']
        except KeyError: baseline_std = BASELINE_STD

        # set baseline mean (default in [-0.5, 0.5] ADC)
        try: baseline_mean_limit = kwargs['baseline_mean']
        except KeyError: baseline_mean_limit = BASELINE_MEAN        

        TrainingSet = EventGenerator(dataset, True, split, shuffle, input_shape, baseline_std, baseline_mean_limit)
        ValidationSet = EventGenerator(dataset, False, split, shuffle, input_shape, baseline_std, baseline_mean_limit)

        return TrainingSet, ValidationSet 

# Generator class for NN sequential model with some additional functionalities
# This website helped tremendously with writing a working example: shorturl.at/fFI09
class EventGenerator(tf.keras.utils.Sequence):

    def __init__(self, dataset, *args) -> None :

        # select files for specific event generator
        unique_events = np.unique([event[:13] for event in os.listdir(os.environ.get('DATA') + dataset)])
        start = 0 if args[0] else int(args[1] * len(unique_events))
        stop = int(args[1] * len(unique_events)) if args[0] else -1

        # specify dataset architecture
        self.__files = unique_events[start : stop]
        self.__shuffle, self.__shape = args[2], args[3]
        self.__std, self.__mean = args[4], args[5]

    # returns one batch of data
    def __getitem__(self, index) -> tuple :

        traces, labels = []
        # TODO add events to above lists

        return (traces, labels)

    # returns the number of batches per epoch
    def __len__(self) -> int :
        return len(self.__files)

     # called by model.fit at the end of each epoch
    def on_epoch_end(self) -> None : 
        self.__shuffle and np.random.shuffle(self.__files)












    

if __name__=="__main__":

    TrainingSet, ValidationSet = DataSetGenerator("second_simulation/tensorflow/signal")