from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
from matplotlib.patches import Rectangle
from matplotlib.colors import BoundaryNorm
from matplotlib.colorbar import ColorbarBase


class EmptyFileError(Exception): pass
class SlidingWindowError(Exception): pass
class EarlyStoppingError(Exception): pass
class SignalError(Exception): pass
class RandomTraceError(Exception): pass

@dataclass
class GLOBAL():
    
    # electronics constants set in offline, see Framework/SDetector/SdSimCalibrationConstants.xml.in
    q_peak                      = 215.934                                       # calibration factor for ADC - VEM
    q_charge                    = 1606.467                                      # calibration factor for integrals

    # Trace details, can be overwritten in __new__ of EventGenerator
    background_frequency        = 4665                                          # frequency of accidental injections / Hz
    single_bin_duration         = 8.3e-9                                        # time length of a single bin, in s                                               
    n_bins                      = 2048                                          # 1 Bin = 8.3 ns, 2048 Bins = ~17. µs
    baseline_std                = 2                                             # two ADC counts, NOT converted here!
    baseline_mean               = 0                                             # gaussian mean of the actual baseline
    real_background             = False                                         # use random traces instead of gaussian baseline
    random_index                = None                                          # this file is used first when creating randoms
    force_inject                = None                                          # whether or not to force injection of muons
    station                     = None                                          # what station to use for random traces

    # trace_opts                  = [q_peak, q_charge, length, sigma, mu, n_injected, downsampling]

    # Generator details, can be overwritten in __new__ of EventGenerator
    split                       = 0.8                                           # Ratio of the training / validation events
    prior                       = 0.5                                           # Probability of a signal event in the data
    seed                        = False                                         # make RNG dice rolls reproducible via seed
    full_trace                  = False                                         # return entire trace instead of sliding window
    downsampling                = False                                         # make UUB traces look like UB ones instead
    
    # Classifier details, can be overwritten in __new__ of EventGenerator
    ignore_low_VEM              = 0                                             # label signals under threshold as background
    ignore_particles            = 0                                             # label traces with n < ignore_particles as bg
    window                      = 120                                           # Length (in bins) of the sliding window
    step                        = 10                                            # Sliding window analysis step size (in bins)
    early_stopping_patience     = 10000                                         # number of batches for early stopping patience
    n_production_traces         = int(1e6)                                      # how many random traces to look at for predictions
    n_ensembles                 = 10                                            # how many networks of same architecture to train

plt.rcParams.update({'font.size': 22})
# plt.rcParams['figure.figsize'] = [20, 10]