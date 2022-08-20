from dataclasses import dataclass
import matplotlib.pyplot as plt

class EmptyFileError(Exception): pass
class SlidingWindowError(Exception): pass
class EarlyStoppingError(Exception): pass

@dataclass
class GLOBAL():

    # Trace details, can be overwritten in __new__ of EventGenerator
    background_frequency        = 4665                                          # frequency of accidental injections / Hz
    single_bin_duration         = 8.3e-9                                        # time length of a single bin, in s                                               
    ADC_to_VEM                  = 215.9                                         # from David's Mail @ 7.06.22 3:30 pm
    n_bins                      = 2048                                          # 1 Bin = 8.3 ns, 2048 Bins = ~17. µs
    baseline_std                = 2                                             # two ADC counts, NOT converted here!
    baseline_mean               = 0                                             # gaussian mean of the actual baseline
    real_background             = False                                         # use random traces instead of gaussian baseline 
    force_inject                = None                                          # whether or not to force injection of muons

    trace_opts                  = [ADC_to_VEM, n_bins, baseline_std, baseline_mean, force_inject, real_background]

    # Generator details, can be overwritten in __new__ of EventGenerator
    split                       = 0.8                                           # Ratio of the training / validation events
    prior                       = 0.5                                           # Probability of a signal event in the data
    seed                        = False                                         # make RNG dice rolls reproducible via seed
    full_trace                  = False                                         # return entire trace instead of sliding window
    
    # Classifier details, can be overwritten in __new__ of EventGenerator
    ignore_low_VEM              = 0                                             # label signals under threshold as background
    window                      = 120                                           # Length (in bins) of the sliding window
    step                        = 10                                            # Sliding window analysis step size (in bins)
    early_stopping_patience     = 1000                                          # number of batches for early stopping patience
    n_production_traces         = int(1e6)                                      # how many random traces to look at for predictions

plt.rcParams.update({'font.size': 22})