#/!usr/bin/python3

import typing
import sys, os
import numpy as np
import tensorflow as tf

# wrapper for easier signal/background event creation
class Event():

    def __new__(cls, label : int, *args, **kwargs) -> typing.Union(Background, Signal):

        if label == 0: return Background(*args, **kwargs)
        if label == 1: return Signal(*args, **kwargs)

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
        for i in range(self.length):
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

        for i in range(window_length,self.length):

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

class Background(Event):

    def __init__(self, ):
        pass

class Signal(Event):

    def __init__(self, ):
        pass

# @dataclass
# class VEMTrace():

#     # TODO overthink structure here, keep long trace baseline?
#     def __init__(self, trace_length : int, mu : float = 0, std : float = 1) -> None :

#         # TODO figure out how this will work
#         self.length = trace_length
#         self.__pmt_1 = None
#         self.__pmt_2 = None
#         self.__pmt_3 = None
    


