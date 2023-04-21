from dataclasses import dataclass
from time import strftime, gmtime
from time import perf_counter_ns
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
from matplotlib.patches import Polygon
from matplotlib.colors import BoundaryNorm
from matplotlib.colorbar import ColorbarBase
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from scipy.stats.mstats import mquantiles
from scipy.optimize import curve_fit
# import seaborn as sns                                                         # causes problems on HTCondor cluster
import tensorflow as tf
import numpy as np
import functools
import warnings
import sys, os
import typing

class EmptyFileError(Exception): pass
class SlidingWindowError(Exception): pass
class EarlyStoppingError(Exception): pass
class SignalError(Exception): pass
class RandomTraceError(Exception): pass
class ElectronicsError(Exception): pass

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    # "font.family": "serif",
    # "font.serif": [],                   # blank entries should cause plots 
    # "font.sans-serif": [],              # to inherit fonts from the document
    # "font.monospace": [],
    # "axes.labelsize": 10,               # LaTeX default is 10pt font.
    # "font.size": 10,
    # "legend.fontsize": 8,               # Make the legend/label fonts 
    # "xtick.labelsize": 8,               # a little smaller
    # "ytick.labelsize": 8,
    "pgf.preamble": "\n".join([ # plots will use this preamble
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[detect-all,locale=DE]{siunitx}",
        ])
    }
plt.rcParams.update(pgf_with_latex)

plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 40
plt.rcParams['figure.figsize'] = [30, 15]

@dataclass
class GLOBAL():
    
    # electronics constants set in offline, see Framework/SDetector/SdSimCalibrationConstants.xml.in
    # for bias factor from TRUE to FPGA vem see Modules/SdReconstruction/SdCalibratorOG/SdCalibrator.xml.in
    q_peak                      = 215.781                                       # OFFLINE calibration factor for ADC - VEM
    q_charge                    = 1593.51                                       # OFFLINE calibration factor for integrals
    q_peak_compatibility        = 163.235                                       # OFFLINE q_peak for downsampled traces

    # Trace details, can be overwritten in __new__ of EventGenerator
    background_frequency        = 4665                                          # frequency of accidental injections / Hz
    single_bin_duration         = 8.3e-9                                        # time length of a single bin, in s                                               
    trace_length                = 2048                                          # 1 Bin = 8.3 ns, 2048 Bins = ~17. µs
    real_background             = True                                          # use random traces instead of gaussian baseline
    random_index                = None                                          # this file is used first when creating randoms
    force_inject                = 0                                             # whether or not to force injection of muons
    station                     = None                                          # what station to use for random traces
    floor_trace                 = True                                          # apply flooring before dividing by q_peak
    is_vem                      = False                                         # provided trace is already calibrated

    # use only for quick checks of performance
    baseline_mean               = 0                                             # gaussian mean of the actual baseline
    baseline_std                = 2                                             # two ADC counts, NOT converted here!

    # Generator details, can be overwritten in __new__ of EventGenerator
    split                       = 0.8                                           # Ratio of the training / validation events
    prior                       = 0.5                                           # Probability of a signal event in the data
    seed                        = False                                         # make RNG dice rolls reproducible via seed
    full_trace                  = False                                         # return entire trace instead of sliding window
    downsampling                = False                                         # make UUB traces look like UB ones instead
    random_phase                = None                                          # the random phase to use for downsampling
    
    # Classifier details, can be overwritten in __new__ of EventGenerator
    ignore_low_VEM              = False                                         # label signals under threshold as background
    ignore_particles            = False                                         # label traces with n < ignore_particles as bg
    particle_type               = ["mu", "e", "ph"]                             # type of particles that count toward cut
    window                      = 120                                           # Length (in bins) of the sliding window
    step                        = 10                                            # Sliding window analysis step size (in bins)
    n_production_traces         = int(1e6)                                      # how many random traces to look at for predictions
    n_ensembles                 = 10                                            # how many networks of same architecture to train
    
    early_stopping_patience     = 7500                                          # number of batches for early stopping patience
    early_stopping_accuracy     = 0.95                                          # activate early stopping above this accuracy

@functools.cache
def get_fit_function(root_path : str, e : int, t : int) -> np.ndarray : 

    c = lambda x, i : x.split("_")[i]
    checksum = lambda x : sum([10*float(c(x,0)), 10*float(c(x,1)), float(c(x,3))/10, float(c(x,4)[:-4])/10])
    ldf_files = np.array(os.listdir(root_path + "FITPARAM/"))[np.argsort([checksum(file) for file in os.listdir(root_path + "FITPARAM/")])]
    ldf_files = [root_path + "FITPARAM/" + ldf_file for ldf_file in ldf_files]
    fit_parameters = np.loadtxt(ldf_files[e * 5 + t])
    
    if "LDF" in root_path or "LTP" in root_path:
        fit_function = lambda x : lateral_distribution_function(x, *fit_parameters)
    else:
        fit_function =  lambda x : lateral_trigger_probability(x, *fit_parameters)

    return fit_function, fit_parameters

def lateral_trigger_probability(x : np.ndarray, efficiency : float, prob_50 : float, scale : float) -> np.ndarray : 

    # # see https://www.sciencedirect.com/sdfe/reader/pii/S0927650511001526/pdf p.7 for full form
    # # To ensure we have a continously differentiable function, we would need to restrict C = 2 * scale
    # near_core = lambda x : 1 / (1 + np.exp(scale * (x - prob_50)))
    # far_core = lambda x : 1 / 2 * np.exp(-efficiency * (x - prob_50))

    # return np.piecewise(x, [x <= prob_50, x > prob_50], [near_core, far_core])

    # # use the same fitfunc as for the lateral distribution function
    return lateral_distribution_function(x, efficiency, prob_50, scale)


def lateral_trigger_probability_error(x : np.ndarray, pcov : np.ndarray, efficiency : float, prob_50 : float, scale : float, C : float = None) -> np.ndarray : 

    # TODO: calculate this when not running in compatibility mode
    return lateral_distribution_function_error(x, pcov, efficiency, prob_50, scale)

def lateral_distribution_function(x : np.ndarray, efficiency : float, prob_50 : float, scale : float) -> np.ndarray :
    return np.clip(efficiency * (1 - 1 / (1 + np.exp(-(x - prob_50)/scale))), 0, 1)

def lateral_distribution_function_error(x : np.ndarray, pcov : np.ndarray, efficiency : float, prob_50 : float, scale : float) -> np.ndarray :

    errors = np.zeros_like(x)

    d_eff = lambda u : lateral_distribution_function(u, efficiency, prob_50, scale) / efficiency
    d_prob_50 = lambda u : efficiency * scale * np.exp(scale * (u + prob_50)) / (np.exp(prob_50 * scale) + np.exp(scale * u))**2
    d_scale = lambda u : efficiency * (prob_50 - u) * np.exp(scale * (prob_50 + u)) / (np.exp(prob_50 * scale) + np.exp(scale * u))**2

    for i, X in enumerate(x):
        gradient = np.array([d_eff(X), d_prob_50(X), d_scale(X)])
        errors[i] += np.sqrt(gradient.T @ pcov @ gradient)

    return errors

def progress_bar(current_step : int, total_steps : int, start_time : int) -> None : 
     
    time_spent = (perf_counter_ns() - start_time) * 1e-9
    ms_per_iteration =  time_spent / (current_step + 1) * 1e3
    elapsed = strftime('%H:%M:%S', gmtime(time_spent))
    eta = strftime('%H:%M:%S', gmtime(time_spent * (total_steps - current_step)/(current_step + 1) ))
    percentage = int((current_step + 1) / total_steps * 100)
    
    print(f"Step {current_step + 1}/{total_steps} | {elapsed} elapsed ||{'-' * (percentage // 5):20}|| {percentage}% -- {ms_per_iteration:.3f} ms/step, ETA: {eta}", end = "\r")