from dataclasses import dataclass
from time import strftime, gmtime
from time import perf_counter_ns
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
from matplotlib.patches import Polygon
from matplotlib.colors import BoundaryNorm
from matplotlib.colorbar import ColorbarBase
from scipy.optimize import curve_fit
import seaborn as sns
import numpy as np
import warnings
import typing

class EmptyFileError(Exception): pass
class SlidingWindowError(Exception): pass
class EarlyStoppingError(Exception): pass
class SignalError(Exception): pass
class RandomTraceError(Exception): pass
class ElectronicsError(Exception): pass

plt.rcParams.update({'font.size': 22})
plt.rcParams['figure.figsize'] = [30, 15]

@dataclass
class GLOBAL():
    
    # electronics constants set in offline, see Framework/SDetector/SdSimCalibrationConstants.xml.in
    q_peak                      = 215.934                                       # OFFLINE calibration factor for ADC - VEM
    q_charge                    = 1606.467                                      # OFFLINE calibration factor for integrals

    # Trace details, can be overwritten in __new__ of EventGenerator
    background_frequency        = 4665                                          # frequency of accidental injections / Hz
    single_bin_duration         = 8.3e-9                                        # time length of a single bin, in s                                               
    trace_length                = 2048                                          # 1 Bin = 8.3 ns, 2048 Bins = ~17. µs
    real_background             = True                                          # use random traces instead of gaussian baseline
    random_index                = None                                          # this file is used first when creating randoms
    force_inject                = None                                          # whether or not to force injection of muons
    station                     = None                                          # what station to use for random traces
    
    # use only for quick checks of performance
    baseline_mean               = 0                                             # gaussian mean of the actual baseline
    baseline_std                = 0 # 2                                            # two ADC counts, NOT converted here!

    # Generator details, can be overwritten in __new__ of EventGenerator
    split                       = 0.8                                           # Ratio of the training / validation events
    prior                       = 0.5                                           # Probability of a signal event in the data
    seed                        = False                                         # make RNG dice rolls reproducible via seed
    full_trace                  = False                                         # return entire trace instead of sliding window
    downsampling                = False                                         # make UUB traces look like UB ones instead
    
    # Classifier details, can be overwritten in __new__ of EventGenerator
    ignore_low_VEM              = False                                         # label signals under threshold as background
    ignore_particles            = False                                         # label traces with n < ignore_particles as bg
    window                      = 120                                           # Length (in bins) of the sliding window
    step                        = 10                                            # Sliding window analysis step size (in bins)
    early_stopping_patience     = 7500                                          # number of batches for early stopping patience
    n_production_traces         = int(1e6)                                      # how many random traces to look at for predictions
    n_ensembles                 = 10                                            # how many networks of same architecture to train

def station_hit_probability(x : np.ndarray, efficiency : float, p50 : float, scale : float) -> np.ndarray:
    return efficiency * (1 - 1 / (1 + np.exp(-scale * (x - p50))))

def progress_bar(current_step : int, total_steps : int, start_time : int) -> None : 
     
    time_spent = (perf_counter_ns() - start_time) * 1e-9
    ms_per_iteration =  time_spent / (current_step + 1) * 1e3
    elapsed = strftime('%H:%M:%S', gmtime(time_spent))
    eta = strftime('%H:%M:%S', gmtime(time_spent * (total_steps - current_step)/(current_step + 1) ))
    percentage = int((current_step + 1) / total_steps * 100)
    
    print(f"Step {current_step + 1}/{total_steps} | {elapsed} elapsed ||{'-' * (percentage // 5):20}|| {percentage}% -- {ms_per_iteration:.3f} ms/step, ETA: {eta}", end = "\r")
    if current_step + 1 == total_steps: print()