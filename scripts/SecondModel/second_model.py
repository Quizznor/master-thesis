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
        
        # comparison to classical triggers: https://arxiv.org/pdf/1111.6764.pdf

        # whether or not T1 trigger would have activated
        self.T1_active = self.check_threshold_trigger(threshold = 1.75)

        # whether or not T2 trigger would have activated
        self.T2_active = self.check_threshold_trigger(threshold = 3.20)

        # whether or not ToT trigger would have activated
        self.ToT_active = False

    # method for checking T1/T2 trigger
    # T1 threshold = 1.75 VEM in all 3 PMTs
    # T2 threshold = 3.20 VEM in all 3 PMTs
    # TODO does this need to be in the same time bin? (i.e. what i wrote below?)
    def check_threshold_trigger(self, threshold : float) -> bool :
        
        # hierarchy doesn't matter, since we need coincident signal anyway
        for i in range(self.length):

            if self.__pmt_1[i] >= threshold:
                if self.__pmt_2[i] >= threshold:
                    if self.__pmt_3[i] >= threshold:
                        return True
                    else: continue
                else: continue
            else: continue
        
        return False

    # method for checking ToT trigger
    # requires 2/3 PMTs have >= 13 bins over 0.2 VEM within ~ 1 μs
    # TODO is this performant? Maybe only evaluate bin when it's added/removed
    def check_time_over_threshold_trigger(self, window_length : int = 120) -> bool :

        # keep <window_length> bins in buffer
        pmt1_window = self.__pmt_1[:window_length]
        pmt2_window = self.__pmt_2[:window_length]
        pmt3_window = self.__pmt_3[:window_length]

        for i in range(window_length, self.length):

            # check if ToT conditions are met
            pmt1_active = len(pmt1_window[pmt1_window > 0.2]) >= 13
            pmt2_active = len(pmt2_window[pmt2_window > 0.2]) >= 13
            pmt3_active = len(pmt3_window[pmt3_window > 0.2]) >= 13

            if pmt1_active == pmt2_active == pmt3_active == True:
                return True

            # overwrite oldest bin and reevaluate
            pmt1_window[i % window_length] = self.__pmt_1[i]
            pmt2_window[i % window_length] = self.__pmt_2[i]
            pmt2_window[i % window_length] = self.__pmt_3[i]

        return False

