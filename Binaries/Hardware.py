from .__config__ import *
from .Signal import *
from .Generator import *
from .Classifier import *

# Wrapper for currently employed station-level triggers (Th1, Th2, ToT, etc.)
# Information on magic numbers comes from Davids Mail on 10.03.22 @ 12:30pm
class HardwareClassifier(Classifier):

    def __init__(self, triggers : list = ["th2", "tot", "totd"], name : str = False) : 
        super().__init__(name or "HardwareClassifier")
        self.triggers = []

        if "th1" in triggers: self.triggers.append(self.Th1)
        if "th2" in triggers: self.triggers.append(self.Th2)
        if "tot" in triggers: self.triggers.append(self.ToT)
        if "totd" in triggers: self.triggers.append(self.ToTd)
        if "mops" in triggers: self.triggers.append(self.MoPS_compatibility)

    def __call__(self, trace : typing.Union[np.ndarray, Trace]) -> bool :

        if isinstance(trace, Trace):

            # check Threshold trigger first, they can receive the entire trace at once
            if self.Th2 in self.triggers or self.Th1 in self.triggers:
                full_trace = np.array([trace.pmt_1, trace.pmt_2, trace.pmt_3])
                
                if self.Th1 in self.triggers and self.Th1(full_trace): return True
                if self.Th2 in self.triggers and self.Th2(full_trace): return True

            # check for rest of the triggers in sliding window analysis
            rest_of_the_triggers = [trigger for trigger in self.triggers if trigger not in [self.Th1, self.Th2]]

            if rest_of_the_triggers == []: return False

            for window in trace:
                for trigger in rest_of_the_triggers:
                    if trigger(window): return True
            else: return False
            
        
        elif isinstance(trace, np.ndarray):
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
        multiplicity  = 2                                                           # number of PMTs that satisfy conditions

        pmt_1, pmt_2, pmt_3 = signal

        # count initial active bins
        pmt1_active = list(pmt_1 > threshold).count(True)
        pmt2_active = list(pmt_2 > threshold).count(True)
        pmt3_active = list(pmt_3 > threshold).count(True)
        ToT_trigger = [pmt1_active >= n_bins, pmt2_active >= n_bins, pmt3_active >= n_bins]

        if ToT_trigger.count(True) >= multiplicity:
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

    @staticmethod
    def plot_performance(ax : plt.axes, x : float = None, y : float = None, x_err : float = None, y_err : float = None) -> None :

        x, xerr, y, yerr, r_th2, r_tot, r_tod = np.loadtxt("/cr/data01/filip/models/HardwareClassifier/ROC_curve/money_plot.csv", unpack = True)

        unscaled_index = np.argmin(abs(x - 3.965828361484564080e-01))

        # special treatment for unscaled q_peak prediction 
        ordinate = np.geomspace(0.01, 1)
        coordinate = ordinate * y[unscaled_index]/x[unscaled_index]
        # coordinate_err = np.sqrt((1/x[unscaled_index]**2 * yerr[unscaled_index]**2 + (y[unscaled_index]/x[unscaled_index]**2)**2 * xerr[unscaled_index]**2) * ordinate)

        # ax.plot(ordinate, coordinate, c = "k", ls = "--", lw = 0.7)
        # ax.errorbar(x[unscaled_index], y[unscaled_index], label = r"Classical triggers, Th, ToT, ToTd", fmt = "-o", markersize = 15, mfc = "w", c = "k")
        ax.errorbar(x[unscaled_index], y[unscaled_index], fmt = "-o", markersize = 15, mfc = "w", c = "k")

        # normal treatment for all other predictions
        scaled_indices = [idx != unscaled_index for idx in range(len(x))]

        ax.plot(x[scaled_indices], r_th2[scaled_indices], c = "k", lw = 1, ls = "--")
        ax.plot(x[scaled_indices], r_tot[scaled_indices], c = "k", lw = 1, ls = ":")
        ax.plot(x[scaled_indices], y[scaled_indices], c = "k", lw = 2)
        ax.fill_between(x[scaled_indices], y[scaled_indices] - yerr[scaled_indices], y[scaled_indices] + yerr[scaled_indices], color = "k", alpha = 0.15)