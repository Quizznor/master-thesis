from dataclasses import dataclass
import numpy as np
import typing
import os

# library of real (mostly) background data from random traces
@dataclass
class Baseline():

    baseline_dir : str = "/cr/tempdata01/filip/iRODS/Background/"               # storage path of the baseline lib
    all_files : np.ndarray = np.asarray(os.listdir(baseline_dir))               # container for all baseline files
    n_files : int = len(all_files)                                              # number of available baseline files

    def __init__(self, index : int) -> typing.NoReturn : 

        these_traces = np.loadtxt(Baseline.baseline_dir + Baseline.all_files[index])
        these_traces = np.split(these_traces, len(these_traces) // 3)
        self.these_traces = np.array([station[:,1:] for station in these_traces])

    # get a random traces of n_station number of stations, starting at start
    def get_baseline(self, start : int, n_stations : int) -> np.ndarray : 
        
        if start + n_stations > len(self.these_traces): raise IndexError
        return self.these_traces[start: start + n_stations]

# library of stray (e.g.) muon signals that are accidentally injected
@dataclass
class Background():

    path : str = "/cr/data01/filip/background/single_pmt.dat"                   # storage path of the background lib
    library : np.ndarray = np.loadtxt(path)                                     # contains injected particle signals
    shape : tuple = library.shape                                               # (number, length) of particles in ^

# Event wrapper for measurements of a SINGLE tank with 3 PMTs
class VEMTrace():

    # Common data for all VEM traces; Some of this can be overwritten in __init__
    background_frequency = 4665                                                 # frequency of accidental injections
    single_bin_duration = 8.3e-9                                                # time length of a single bin, in s                                               
    ADC_to_VEM_factor = 215.9                                                   # from David's Mail @ 7.06.22 3:30 pm
    trace_length = 2048                                                         # 1 Bin = 8.3 ns, 2048 Bins = ~17. µs
    baseline_std = 2                                                            # two FAD counts, NOT converted here!
    baseline_limits = [-2, 2]                                                   # same goes for (the limits) of means
    baseline_mean = 0                                                           # not set here, actual baseline value                                                                   

    # metadata regarding shower origin, energy and so on
    StationID  = -1                                                             # set all these values to nonsensical
    Energy     = -1                                                             # numbers in the beginning for easier 
    SPDistance = -1                                                             # distinguishing between actual event 
    Zenith     = -1                                                             # and background traces down the line
    # TODO add timing information here? 
    # Might be needed for CDAS triggers ...

    def __init__(self, trace_data : np.ndarray = None, baseline_data : np.ndarray = None, **kwargs) -> typing.NoReturn :

        r'''
        :trace_data ``tuple``: tuple with individual pmt data in each entry of the tuple. If None, background trace is raised

        :Keyword arguments:
            * *ADC_to_VEM* (``float``) -- ADC to VEM conversion factor, important for ub <-> uub
            * *n_bins* (``int``) -- generate a baseline with <trace_length> bins
            * *force_inject* (``int``) -- force the injection of <force_inject> pbackground particles
            * *sigma* (``float``) -- baseline std in ADC counts, ignored if real_background = True
            * *mu_lim* (``list``) -- mean ADC level limit [low, high] in ADC counts), ignored if real_background = True
            * *mu* (``float``) -- the actual value of the trace background baseline, ignored if real_background = True

        Initialization fails if metadata doesn't match for the different PMTs
        or the baseline length is too short. In both cases a ValueError is raised 
        '''

        # Change VEM trace defaults (if desired)
        self.ADC_to_VEM_factor = kwargs.get("ADC_to_VEM", VEMTrace.ADC_to_VEM_factor)
        self.trace_length = kwargs.get("n_bins", VEMTrace.trace_length)
        self.baseline_limits = kwargs.get("mu_lim", VEMTrace.baseline_limits)
        self.baseline_mean = kwargs.get("mu", np.random.uniform(*VEMTrace.baseline_limits) / self.ADC_to_VEM_factor)
        self.baseline_std = kwargs.get("sigma", VEMTrace.baseline_std / self.ADC_to_VEM_factor)

        # Create a baseline for each PMT
        if baseline_data is not None:
            self.Baseline = baseline_data
        else:
            self.Baseline = np.random.normal(self.baseline_mean, self.baseline_std, (3, self.trace_length))

        # Create container for Signal
        self.Signal = None

        if trace_data is not None:

            assert self.trace_length > trace_data.shape[1], "signal size exceeds trace length"

            # group trace information first
            station_ids = set(trace_data[:,0])
            sp_distances = set(trace_data[:,1])
            energies = set(trace_data[:,2])
            zeniths = set(trace_data[:,3])
            trace_data = trace_data[:,4:]

            # assert that metadata looks the same for all three PMTs
            for metadata in [station_ids, sp_distances, energies, zeniths]:
                assert len(metadata) == 1, "Metadata between PMTs doesn't match"

            # assert that pmt traces all have the same length
            assert len(trace_data[0]) == len(trace_data[1]) == len(trace_data[2]), "PMTs have different bin lengths"

            self.StationID = next(iter(station_ids))                            # the ID of the station in question
            self.SPDistance = next(iter(sp_distances))                          # the distance from the shower core
            self.Energy = next(iter(energies))                                  # energy of the shower of this signal
            self.Zenith = next(iter(zeniths))                                   # zenith of the shower of this signal
            self.Signal = np.zeros((3, self.trace_length))                      # container to store signal trace only

            self._sig_injected_at = np.random.randint(-self.trace_length, -trace_data.shape[1])
            self._sig_stopped_at  = self._sig_injected_at + trace_data.shape[1] + self.trace_length


            for i, pmt in enumerate(trace_data):
                self.Signal[i][self._sig_injected_at : self._sig_injected_at + trace_data.shape[1]] += pmt

            self._sig_injected_at += self.trace_length

        # Accidentally inject background particles
        self.Background = None

        n_inject = np.random.poisson( self.background_frequency * self.single_bin_duration * self.trace_length )
        self.n_injected = n_inject if np.isnan(kwargs.get("force_inject", n_inject)) else kwargs.get("force_inject", n_inject)

        if self.n_injected != 0:
            self.Background = np.zeros((3, self.trace_length))
            self._bkg_injected_at, self._bkg_stopped_at = [], []

            for i in range(self.n_injected):
                particle = np.random.randint(0, Background.shape[0])
                background_particle = Background.library[particle]

                injected_at = np.random.randint(-self.trace_length, -Background.shape[1])
                self._bkg_stopped_at.append( injected_at + len(background_particle) + self.trace_length )
                self._bkg_injected_at.append( injected_at + self.trace_length )

                for j in range(3):
                    self.Background[j][injected_at : injected_at + Background.shape[1]] += background_particle
    

        # Add all different components (Baseline, Injected, Signal) together
        self.pmt_1, self.pmt_2, self.pmt_3 = self.Baseline
        for Component in [self.Signal, self.Background]:
            if Component is not None:
                self.pmt_1 += Component[0]
                self.pmt_2 += Component[1]
                self.pmt_3 += Component[2]

        # and convert everything from ADC to VEM counts (if desired)
        self.pmt_1 = self.convert_to_VEM(self.pmt_1)
        self.pmt_2 = self.convert_to_VEM(self.pmt_2)
        self.pmt_3 = self.convert_to_VEM(self.pmt_3)    

    # return a label and sector of the whole trace, specify window length and starting position
    def get_trace_window(self, start_bin : int, window_length : int, threshold : float = None) -> tuple :

        assert start_bin + window_length <= self.trace_length, "trace sector exceeds the whole trace length"

        start_at, stop_at, label = start_bin, start_bin + window_length, 0
        cut = lambda array : array[start_at : stop_at]
        trace_window = np.array([cut(self.pmt_1), cut(self.pmt_2), cut(self.pmt_3)])

        # check whether signal and window frame overlap AND signal exceeds <threshold> VEM
        try:

            signal = range(self._sig_injected_at, self._sig_stopped_at)
            window = range(start_at, stop_at)

            signal_has_overlap = len(range(max(signal[0], window[0]), min(signal[-1], window[-1]) + 1 )) == 0
            
            if threshold is not None:

                if VEMTrace.integrate(trace_window) >= threshold:
                    label = 1 if signal_has_overlap else 0
                else:
                    label = 0
            else:
                label = 1 if signal_has_overlap else 0

        except AttributeError:
            label = 0

        return label, trace_window

    # return the number of bins containing a (signal, background) of a given window
    def get_n_signal_background_bins(self, index : int, window_length : int) -> tuple : 

        try:
            n_bkg_bins = 0
            n_bkg_bins += len(np.unique(np.nonzero(self.Injected[:, index : index + window_length])[1]))
        except (AttributeError, TypeError):
            pass

        try:
            n_sig_bins = len(np.unique(np.nonzero(self.Signal[:, index : index + window_length])[1]))
        except (AttributeError, TypeError):
            n_sig_bins = 0

        return n_sig_bins, n_bkg_bins

    # convert array of FADC counts to array of VEM counts
    def convert_to_VEM(self, signal : np.ndarray) -> np.ndarray :
        return np.floor(signal) / self.ADC_to_VEM_factor

    # return a specific component of the trace (signal, background, baseline)
    def get_component(self, key : str) -> np.ndarray :

        components = {"signal" : self.Signal,
                      "background" : self.Background,
                      "baseline" : self.Baseline}

        if components[key] is not None:
            print("\nWARNING: COMPONENT TRACES ARE GIVEN IN ADC COUNTS, NOT VEM!\n")
            return components[key]
        else:
            raise AttributeError(f"VEM trace does not have a component: {key}")

    # return the integrated signal of a trace window
    @staticmethod
    def integrate(window : np.ndarray) -> float : 
        return np.round(np.mean(np.sum(window, axis = 1)),2)

    # plot the trace in whatever figure plt.gca() points to
    def plot(self, accumulate : bool = True) -> typing.NoReturn :

        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size': 22})

        plt.plot(range(self.trace_length), self.pmt_1, label = "PMT #1")
        plt.plot(range(self.trace_length), self.pmt_2, label = "PMT #2")
        plt.plot(range(self.trace_length), self.pmt_3, label = "PMT #3")

        try:
            plt.axvline(self._sig_injected_at, ls = "--", c = "g")
            plt.axvline(self._sig_stopped_at, ls = "--", c = "r")
        except AttributeError:
            pass

        # plt.xlim(0,self.trace_length)
        plt.xlabel("Time bin / 8.3 ns")
        plt.ylabel("Signal / VEM")
        plt.legend()

        accumulate and plt.show()