#/!usr/bin/python3

import typing
import sys, os
import numpy as np
import tensorflow as tf

class VEMTrace():

    # TODO overthink structure here, keep long trace baseline?
    def __init__(self, trace_length : int, mu : float = 0, std : float = 1) -> None :

        self.length = trace_length
        self.__pmt_1 = None
        self.__pmt_2 = None
        self.__pmt_3 = None

class Trigger():

    # T1 threshold = 1.75 VEM in all 3 PMTs
    def T1_trigger(self, Trace : VEMTrace):
        return self.absolute_threshold_trigger(Trace, 1.75)

    # T2 threshold = 3.20 VEM in all 3 PMTs
    def T2_trigger(self, Trace : VEMTrace):
        return self.absolute_threshold_trigger(Trace, 3.20)

    # requires 2/3 PMTs have >= 13 bins over 0.2 VEM within ~ 1 μs
    # TODO is this performant enough? I trimmed it down as good as I can
    def ToT_trigger(self, Trace : VEMTrace, window_length : int = 120, threshold : float = 0.2) -> bool :

        # count initial active bins
        pmt1_active = len(Trace.__pmt_1[:window_length][Trace.__pmt_1[:window_length] > threshold])
        pmt2_active = len(Trace.__pmt_2[:window_length][Trace.__pmt_2[:window_length] > threshold])
        pmt3_active = len(Trace.__pmt_3[:window_length][Trace.__pmt_3[:window_length] > threshold])

        for i in range(window_length,Trace.length):

            # check if ToT conditions are met
            ToT_trigger = [pmt1_active >= 13, pmt2_active >= 13, pmt3_active >= 13]

            if ToT_trigger.count(True) >= 2:
                return True

            # overwrite oldest bin and reevaluate
            pmt1_active += self.updated_bin_count(i, Trace.__pmt_1)
            pmt2_active += self.updated_bin_count(i, Trace.__pmt_2)
            pmt3_active += self.updated_bin_count(i, Trace.__pmt_3)

        return False

    # Helper for ToT trigger
    @staticmethod
    def updated_bin_count(index : int, array: np.ndarray, window_length : int = 120, threshold : float = 0.2) -> int :

        # is new bin active?
        if array[index] >= threshold:
            updated_bin_count = 1
        else:
            updated_bin_count = 0

        # was old bin active?
        if array[index - window_length] >= threshold:
            updated_bin_count -= 1

        return updated_bin_count

    # Check for coincident signals in all three photomultipliers
    @staticmethod
    def absolute_threshold_trigger(Trace, threshold : float) -> bool :
        
        # hierarchy doesn't (shouldn't?) matter, since we need coincident signal anyway
        for i in range(Trace.length):

            if Trace.__pmt_1[i] >= threshold:
                if Trace.__pmt_2[i] >= threshold:
                    if Trace.__pmt_3[i] >= threshold:
                        return True
                    else: continue
                else: continue
            else: continue
        
        return False
