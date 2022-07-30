from dataclasses import dataclass
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os, sys
import random
import typing

class EmptyFileError(Exception): pass

@dataclass
class GLOBAL():

    # Trace details, can be overwritten in __new__ of EventGenerator
    background_frequency        = 4665                                          # frequency of accidental injections
    single_bin_duration         = 8.3e-9                                        # time length of a single bin, in s                                               
    ADC_to_VEM                  = 215.9                                         # from David's Mail @ 7.06.22 3:30 pm
    n_bins                      = 2048                                          # 1 Bin = 8.3 ns, 2048 Bins = ~17. µs
    baseline_std                = 2                                             # two ADC counts, NOT converted here!
    baseline_mean               = 0                                             # gaussian mean of the actual baseline
    real_background             = False                                         # use random traces instead of gaussian baseline 
    force_inject                = False                                         # whether or not to force injection of muons

    # Generator details, can be overwritten in __new__ of EventGenerator
    split                       = 0.8                                           # Ratio of the training / validation events
    seed                        = False                                         # make RNG dice rolls reproducible via seed
    prior                       = 0.5                                           # Probability of a signal event in the data
    
    # Classifier details, can be overwritten in __new__ of EventGenerator
    ignore_low_VEM              = None                                          # intentionally mislabel low VEM trace windows
    window                      = 120                                           # Length (in bins) of the sliding window
    step                        = 10                                            # Sliding window analysis step size (in bins)

from TriggerStudyBinaries_v2.Generator import EventGenerator


# from TriggerStudyBinaries_v2.Classifier import TriggerClassifier, NNClassifier
# from TriggerStudyBinaries_v2.Signal import VEMTrace, Background, Baseline