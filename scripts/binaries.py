import numpy as np
import typing, os

# baseline shape defaults 1 VEM_peak = 61.75 ADC !
# see David's mail from 08.02 for info on magic numbers

BASELINE_LENGTH = 20000                 # number of bins in baseline
BASELINE_STD = 0.5 / 61.75              # baseline std in VEM counts
BASELINE_MEAN = [-8.097e-3, +8.097e-3]  # mean ADC level limits [low, high]

# Event wrapper for measurements of a SINGLE tank with 3 PMTs
class VEMTrace():

    r'''
    :Keyword arguments:
        * *trace_length* (``int``) -- number of bins in the trace
        * *baseline_std* (``float``) -- baseline std in VEM counts
        * *baseline_mean* (``list``) -- mean ADC level limit [low, high]
        * *signal* (``str``) -- location (relative to $DATA) of signal file
    '''

    def __init__(self, label : str, **kwargs) -> None:

        self.label = label              # whether trace is signal or background

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
            vem_signals = np.loadtxt(os.environ.get('DATA') + kwargs['signal'])
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

if __name__=="__main__":

    test = VEMTrace("signal", signal="second_simulation/tensorflow/signal/DAT762042_00_station-12.csv")