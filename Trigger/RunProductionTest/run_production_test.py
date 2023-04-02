#!/usr/bin/env python3

import sys
import numpy as np
import sys, os
import random

class EmptyFileError(Exception): pass
class SlidingWindowError(Exception): pass
class EarlyStoppingError(Exception): pass
class SignalError(Exception): pass
class RandomTraceError(Exception): pass

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
    window                      = 120                                           # Length (in bins) of the sliding window
    step                        = 10                                            # Sliding window analysis step size (in bins)
    early_stopping_patience     = 10000                                         # number of batches for early stopping patience
    n_production_traces         = int(1e6)                                      # how many random traces to look at for predictions
    n_ensembles                 = 10                                            # how many networks of same architecture to train


class RandomTrace():

    baseline_dir : str = "/cr/tempdata01/filip/iRODS/"                              # storage path of the station folders
    # all_files : np.ndarray = np.asarray(os.listdir(baseline_dir))                   # container for all baseline files
    # all_n_files : int = len(all_files)                                              # number of available baseline files

    def __init__(self, station : str = None, index : int = None) -> None : 

        ## (HOPEFULLY) TEMPORARILY FIXED TO NURIA/LO_QUI_DON DUE TO BAD FLUCTUATIONS IN OTHER STATIONS
        self.station = random.choice(["nuria"]) if station is None else station.lower()
        self.index = index

        all_files = np.asarray(os.listdir(RandomTrace.baseline_dir + self.station)) # container for all baseline files
        self.all_n_files = len(all_files)                                           # number of available baseline files

        self.__current_traces = 0                                                   # number of traces already raised

        if index is None:
            self.random_file = all_files[np.random.randint(self.all_n_files)]
        else:
            try:
                self.random_file = all_files[index]
            except IndexError:
                raise RandomTraceError

        print(f"[INFO] -- LOADING {station}: {self.random_file}... ", end = "")

        these_traces = np.loadtxt(RandomTrace.baseline_dir + self.station + "/" + self.random_file)

        # IF YOU WANT TO USE DAY AVERAGE FROM ONLINE ESTIMATE #########################################
        # values come from $TMPDATA/iRODS/MonitoringData/read_monitoring_data.ipynb -> monitoring files
        if "nuria" in self.station:
            self.q_peak = [180.23, 182.52, 169.56]
            self.q_charge = [3380.59, 3508.69, 3158.88]
            print("DONE")
        elif "lo_qui_don" in self.station:
            # self.q_peak = [164.79, 163.49, 174.71]
            self.q_peak = [163.79, 162.49, 173.71]
            self.q_charge = [2846.67, 2809.48, 2979.65]
            print("DONE")
        elif "jaco" in self.station:
            self.q_peak = [189.56, 156.48, 168.20]
            self.q_charge = [3162.34, 2641.25, 2840.97]
            print("DONE")
        elif "peru" in self.station:
            self.q_peak = [164.02, 176.88, 167.37]
            self.q_charge = [2761.37, 3007.72, 2734.63]
            print("DONE")
        else:
            print("Station not found! THIS SHOULD NOT HAPPEN")
            self.q_peak = [GLOBAL.q_peak for i in range(3)]
            self.q_charge = [GLOBAL.q_charge for i in range(3)]

       

        self._these_traces = np.split(these_traces, len(these_traces) // 3)         # group random traces by pmt


    # get random traces for a single stations
    def get(self) -> np.ndarray : 
        
        try:                                                                        # update pointer after loading
            self.__current_traces += 1

            return self.q_peak, self.q_charge, self._these_traces[self.__current_traces]
        
        except IndexError:                                                          # reload buffer on overflow

            try: self.__init__(station = self.station, index = self.index + 1)
            except TypeError: self.__init__(station = self.station)

            return self.get()


class HardwareClassifier():

    def __init__(self, triggers : list = ["th2", "tot", "totd", "mops"], name : str = False) : 
        self.triggers = []

        if "th" in triggers: self.triggers.append(self.Th1)
        if "th2" in triggers: self.triggers.append(self.Th2)
        if "tot" in triggers: self.triggers.append(self.ToT)
        if "totd" in triggers: self.triggers.append(self.ToTd)
        if "mops" in triggers: self.triggers.append(self.MoPS_compatibility)

    def __call__(self, trace : np.ndarray) -> bool : 
        
        try:
            for trigger in self.triggers:
                if trigger(trace): return True
            else: return False

        except ValueError:
            return np.array([self.__call__(t) for t in trace])

    def Th1(self, signal) -> bool : 
        return self.Th(signal, 1.7)

    def Th2(self, signal) -> bool : 
        return self.Th(signal, 3.2)

    # method to check for (coincident) absolute signal threshold
    @staticmethod
    def Th(signal : np.ndarray, threshold : float) -> bool : 

        # Threshold of 3.2 immediately gets promoted to T2
        # Threshold of 1.75 if a T3 has already been issued
        pmt_1, pmt_2, pmt_3 = signal

        # hierarchy doesn't (shouldn't?) matter
        for i in range(signal.shape[1]):
            if pmt_1[i] > threshold:
                if pmt_2[i] > threshold:
                    if pmt_3[i] > threshold:
                        return True
                    else: continue
                else: continue
            else: continue
        
        return False

    # method to check for elevated baseline threshold trigger
    @staticmethod
    def ToT(signal : np.ndarray) -> bool : 

        threshold     = 0.2                                                         # bins above this threshold are 'active'
        n_bins        = 13                                                          # minimum number of bins above threshold

        pmt_1, pmt_2, pmt_3 = signal

        # count initial active bins
        pmt1_active = list(pmt_1 > threshold).count(True)
        pmt2_active = list(pmt_2 > threshold).count(True)
        pmt3_active = list(pmt_3 > threshold).count(True)
        ToT_trigger = [pmt1_active >= n_bins, pmt2_active >= n_bins, pmt3_active >= n_bins]

        if ToT_trigger.count(True) >= 2:
            return True
        else:
            return False

    # method to check for elevated baseline of deconvoluted signal
    # note that this only ever gets applied to UB-like traces, with 25 ns binning
    @staticmethod
    def ToTd(signal : np.ndarray) -> bool : 

        # for information on this see GAP note 2018-01
        dt      = 25                                                                # UB bin width
        tau     = 67                                                                # decay constant
        decay   = np.exp(-dt/tau)                                                   # decay term
        deconvoluted_trace = []

        for pmt in signal:
            deconvoluted_pmt = [(pmt[i] - pmt[i-1] * decay)/(1 - decay) for i in range(1,len(pmt))]
            deconvoluted_trace.append(deconvoluted_pmt)
 
        return HardwareClassifier.ToT(np.array(deconvoluted_trace))
    
    # count the number of rising flanks in a trace
    # WARNING: THIS IMPLEMENTATION IS NOT CORRECT!
    @staticmethod
    def MoPS_compatibility(signal : np.ndarray) -> bool : 

        # assumes we're fed 120 bin trace windows
        y_min, y_max = 3, 31                                                        # only accept y_rise in this range
        min_slope_length = 2                                                        # minimum number of increasing bins
        min_occupancy = 4                                                           # positive steps per PMT for trigger
        pmt_multiplicity = 2                                                        # required multiplicity for PMTs
        pmt_active_counter = 0                                                      # actual multiplicity for PMTs
        integral_check = 75 * 120/250                                               # integral must pass this threshold


        for pmt in signal:

            # check for the modified integral first, computationally easier
            if sum(pmt) * GLOBAL.q_peak <= integral_check: continue
            
            occupancy = 0
            positive_steps = np.nonzero(np.diff(pmt) > 0)[0]
            steps_isolated = np.split(positive_steps, np.where(np.diff(positive_steps) != 1)[0] + 1)
            candidate_flanks = [step for step in steps_isolated if len(step) >= min_slope_length]
            candidate_flanks = [np.append(flank, flank[-1] + 1) for flank in candidate_flanks]

            for i, flank in enumerate(candidate_flanks):

                # adjust searching area after encountering a positive flank
                total_y_rise = (pmt[flank[-1]] - pmt[flank[0]]) * GLOBAL.q_peak
                n_continue_at_index = flank[-1] + max(0, int(np.log2(total_y_rise) - 2))

                for j, consecutive_flank in enumerate(candidate_flanks[i + 1:], 0):

                    if n_continue_at_index <= consecutive_flank[0]: break                                                       # no overlap, no need to do anything
                    elif consecutive_flank[0] < n_continue_at_index <= consecutive_flank[-1]:                                   # partial overlap, adjust next flank
                        overlap_index = np.argmin(np.abs(consecutive_flank - n_continue_at_index))
                        candidate_flanks[i + j] = candidate_flanks[i + j][overlap_index:]
                        if len(candidate_flanks[i + j]) < min_slope_length: _ = candidate_flanks.pop(i + j)
                        break

                    elif consecutive_flank[-1] < n_continue_at_index:                                                           # complete overlap, discard next flank
                        _ = candidate_flanks.pop(i + j)

                if total_y_rise > y_min and total_y_rise < y_max: occupancy += 1
            if occupancy > min_occupancy: pmt_active_counter += 1
            
        return pmt_active_counter >= pmt_multiplicity


def apply_downsampling(pmt : np.ndarray, random_phase : int) -> np.ndarray :

    n_bins_uub      = (len(pmt) // 3) * 3               # original trace length
    n_bins_ub       = n_bins_uub // 3                   # downsampled trace length
    sampled_trace   = np.zeros(n_bins_ub)               # downsampled trace container
    kADCsaturation = 4095                               # HG max saturation bin value

    # ensure downsampling works as intended
    # cuts away (at most) the last two bins
    if len(pmt) % 3 != 0: pmt = pmt[0 : -(len(pmt) % 3)]

    # see Framework/SDetector/UUBDownsampleFilter.h in Offline main branch for more information
    kFirCoefficients = [ 5, 0, 12, 22, 0, -61, -96, 0, 256, 551, 681, 551, 256, 0, -96, -61, 0, 22, 12, 0, 5 ]
    buffer_length = int(0.5 * len(kFirCoefficients))
    kFirNormalizationBitShift = 11

    temp = np.zeros(n_bins_uub + len(kFirCoefficients))

    temp[0 : buffer_length] = pmt[:: -1][-buffer_length - 1 : -1]
    temp[-buffer_length - 1: -1] = pmt[:: -1][0 : buffer_length]
    temp[buffer_length : -buffer_length - 1] = pmt

    # perform downsampling
    for j, coeff in enumerate(kFirCoefficients):
        sampled_trace += [temp[k + j] * coeff for k in range(random_phase, n_bins_uub, 3)]

    # clipping and bitshifting
    sampled_trace = [int(adc) >> kFirNormalizationBitShift for adc in sampled_trace]

    # Simulate saturation of PMTs at 4095 ADC counts ~ 19 VEM <- same for HG/LG? I doubt it
    return np.clip(np.array(sampled_trace), a_min = -10, a_max =  kADCsaturation)

station = "nuria"

i = int(sys.argv[1])
trace_duration = GLOBAL.n_bins * GLOBAL.single_bin_duration

window_start = range(0, 682 - 120, 10)
window_stop = range(120, 682, 10)
Trigger = HardwareClassifier()

Buffer = RandomTrace(station = station, index = i)
n_traces = len(Buffer._these_traces)
duration = n_traces * trace_duration

percentages = [-16, -14, -13, -12, -11, -8, -6, -4, -2, -1, 0, 1, 2, 4, 8, 16]
# percentages += list(np.arange(-50, -25, 5)) + list(np.arange(20, 101, 5))
percentages = np.array(percentages) * 1e-2

# # increments = np.array([-4, -2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20])

for percentage in percentages:
# for increment in increments:

    pm = "p" if percentage >= 0 else "m"
    # pm = "p" if increment >= 0 else "m"
    n_trigger = n_th = n_tot = n_totd = 0

    for j, trace in enumerate(Buffer._these_traces):

        # apply downsampling to trace
        downsampled_trace = np.array([(apply_downsampling(trace[k], np.random.randint(3))) / (Buffer.q_peak[k] * (1 + percentage)) for k in range(3)])
        # downsampled_trace = np.array([(apply_downsampling(trace)[k] + increment) / Buffer.q_peak[k] for k in range(3)])

        # split trigger procedure up into different chunks due to performance
        for (start, stop) in zip(window_start, window_stop):
            window = downsampled_trace[:, start : stop]
            if Trigger(window):
                n_trigger += 1
                break

    perc_str = str(int(percentage * 100)).replace('-','')
    print(f"/cr/users/filip/Trigger/RunProductionTest/trigger_output_with_mops/{station}/{station}_all_triggers_with_mops_{pm}{perc_str}.csv")

    with open(f"/cr/users/filip/Trigger/RunProductionTest/trigger_output_with_mops/{station}/{station}_all_triggers_with_mops_{pm}{perc_str}.csv", "a") as f:
        f.write(f"{Buffer.random_file} {n_traces} {duration} {n_trigger}\n")

    # inc_str = str(increment).replace('-','')
    # with open(f"/cr/users/filip/Trigger/RunProductionTest/trigger_output/{station}/trace_increment/{station}_all_triggers_{pm}{inc_str}.csv", "a") as f:
    #     f.write(f"{Buffer.random_file} {n_traces} {duration} {n_trigger} {n_th} {n_tot} {n_totd}\n")