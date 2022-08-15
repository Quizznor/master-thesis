from numba.experimental import jitclass
from numba import njit, vectorize
import numpy as np
import os

from .__config__ import *

# container for simulated signal
class Signal():

    def __init__(self, pmt_data : np.ndarray, trace_length : int) -> None :

        assert len(pmt_data[0]) == len(pmt_data[1]) == len(pmt_data[2]), "PMTs have differing signal length"

        # group trace information first
        station_ids = set(pmt_data[:,0])
        sp_distances = set(pmt_data[:,1])
        energies = set(pmt_data[:,2])
        zeniths = set(pmt_data[:,3])
        pmt_data = pmt_data[:,4:]

        assert trace_length > len(pmt_data[0]), "signal size exceeds trace length"

        # assert that metadata looks the same for all three PMTs
        for metadata in [station_ids, sp_distances, energies, zeniths]:
            assert len(metadata) == 1, "Metadata between PMTs doesn't match"

        self.StationID = int(next(iter(station_ids)))                               # the ID of the station in question
        self.SPDistance = int(next(iter(sp_distances)))                             # the distance from the shower core
        self.Energy = next(iter(energies))                                          # energy of the shower of this signal
        self.Zenith = next(iter(zeniths))                                           # zenith of the shower of this signal

        self.Signal = np.zeros((3, trace_length))
        self.signal_start = np.random.randint(0, trace_length - len(pmt_data[0]))
        self.signal_end = self.signal_start + len(pmt_data[0])

        for i, PMT in enumerate(pmt_data):
            self.Signal[i][self.signal_start : self.signal_end] += PMT

# container for the combined trace
class Trace(Signal):

    def __init__(self, trace_options : list, baseline_data : np.ndarray, signal_data : tuple = None) :

        self.ADC_to_VEM = trace_options[0]
        self.length = trace_options[1]
        self.sigma = trace_options[2]
        self.mu = trace_options[3]

        if trace_options[4] is not None:
            self.inject = trace_options[4]
        else: self.inject = self.poisson()

        if signal_data is not None: 
            super().__init__(signal_data, self.length)
            self.has_signal = True
        else:
            self.Signal = None
            self.has_signal = False

        if self.inject:
            self.injections_start, self.injections_end, self.Injected = InjectedBackground(self.inject, self.length)
            self.has_accidentals = True
        else: 
            self.Injected = None
            self.has_accidentals = False

        self.Baseline = baseline_data

        if self.has_accidentals and self.has_signal:
            self.pmt_1, self.pmt_2, self.pmt_3 = self.convert_to_VEM ( self.Baseline + self.Signal + self.Injected, self.ADC_to_VEM )
        elif self.has_accidentals:
            self.pmt_1, self.pmt_2, self.pmt_3 = self.convert_to_VEM ( self.Baseline + self.Injected, self.ADC_to_VEM )
        elif self.has_signal:
            self.pmt_1, self.pmt_2, self.pmt_3 = self.convert_to_VEM ( self.Baseline + self.Signal, self.ADC_to_VEM )
        else: self.pmt_1, self.pmt_2, self.pmt_3 = self.convert_to_VEM ( self.Baseline, self.ADC_to_VEM )

    # poissonian for background injection
    # jitted function for performance
    def poisson(self) -> int :

        f = GLOBAL.background_frequency
        t_bin = GLOBAL.single_bin_duration
        n_bin = self.length

        return self._poisson(f, t_bin, n_bin)

    # extract pmt data plus label for a given trace window 
    # semi-jitted function for performance
    def get_trace_window(self, window : tuple) -> tuple : 

        pmt_1, pmt_2, pmt_3 = self._cut(self.pmt_1, self.pmt_2, self.pmt_3, window)
        label = self.calculate_signal_overlap(window)

        if np.array([pmt_1, pmt_2, pmt_3]).shape != (3,120):
            
            print(window, self.signal_start, self.signal_end)
            raise StopIteration

        return np.array([pmt_1, pmt_2, pmt_3]), label

    # calculate number of bins of signal in sliding window
    # jitted function for performance
    def calculate_signal_overlap(self, window : tuple) -> int :
        
        if self.has_signal: 
            return self._calculate_signal_overlap((self.signal_start, self.signal_end), window)
        else: return 0

    @staticmethod
    @njit
    # return the mean of integrated PMT signals for a given window
    def integrate(window : np.ndarray) -> float : 
        return np.mean(np.sum(window, axis = 1))

    @staticmethod
    @njit
    # convert from ADC counts to VEM 
    def convert_to_VEM(signal : np.ndarray, ADC_to_VEM : float) -> np.ndarray :
        return np.floor(signal) / ADC_to_VEM

    # wrapper for pretty printing
    def __repr__(self) -> str :

        reduce_by = 30
        trace = list(" " * (self.length // reduce_by))

        # indicate background
        if self.has_accidentals:
            for start, stop in zip(self.injections_start, self.injections_end):
                start, stop = start // reduce_by, stop // reduce_by - 1

                trace[start] = "b"

                for signal in range(start + 1, stop):
                    trace[signal] = "-"

                trace[stop] = "b"

        # indicate signal
        if self.has_signal:
            start, stop = self.signal_start // reduce_by, self.signal_end // reduce_by - 1

            trace[start] = "S" if trace[start] == " " else "X"

            for signal in range(start + 1, stop):
                trace[signal] = "=" if trace[signal] == " " else "X"

            trace[stop] = "S" if trace[stop] == " " else "X"

            metadata = f" {self.Energy:.4e} eV @ {self.SPDistance} m from core   "

        else: metadata = " Background trace                     "
        
        return "||" + "".join(trace) + "||" + metadata

    # wrapper for plotting a trace
    def __plot__(self) -> None :

        x = range(self.length)

        plt.plot(x, self.pmt_1, c = "green", label = "PMT #1", lw = 0.5)
        plt.plot(x, self.pmt_2, c = "orange", label = "PMT #2", lw = 0.5)
        plt.plot(x, self.pmt_3, c = "steelblue", label = "PMT #3", lw = 0.5)

        if self.has_signal:
            plt.axvline(self.signal_start, ls = "--", c = "red", lw = 2)
            plt.axvline(self.signal_end, ls = "--", c = "red", lw = 2)

        if self.has_accidentals:
            for start, stop in zip(self.injections_start, self.injections_end):
                plt.axvline(start, ls = "--", c = "gray")
                plt.axvline(stop, ls = "--", c = "gray")

        plt.ylabel("Signal strength / VEM")
        plt.xlabel("Bin / 8.3 ns")
        plt.legend()
        plt.show()

    # helper functions for numba #######################
    @staticmethod
    @njit
    def _poisson(f : float, t_bin : float, n_bin : int) -> int :
        return np.random.poisson( f * t_bin * n_bin )

    @staticmethod
    @njit
    def _calculate_signal_overlap(signal : tuple, window : tuple) -> int :
        return len(range(max(window[0], signal[0]), min(window[-1], signal[1])))

    @staticmethod
    @njit
    def _cut(pmt_1 : np.ndarray, pmt_2 : np.ndarray, pmt_3 : np.ndarray, window : tuple) -> tuple :
        
        start, stop = window
        return (pmt_1[start : stop], pmt_2[start : stop], pmt_3[start : stop])
    ####################################################

# container for reading signal files
class SignalBatch():

    def __new__(self, trace_file : str) -> tuple:

        if not os.path.getsize(trace_file): raise EmptyFileError

        with open(trace_file, "r") as file:
                signal = [np.array([float(x) for x in line.split()]) for line in file.readlines()]

        for station in range(0, len(signal) // 3, 3):
            yield np.array([signal[station], signal[station + 1], signal[station + 2]]) 

# container for gaussian baseline
class Baseline():

    def __new__(self, mu : float, sigma : float, length : int) -> np.ndarray :
        return np.random.normal(mu, sigma, (3, length))

# container for random traces
# @jitclass # ?
class RandomTrace():

    baseline_dir : str = "/cr/tempdata01/filip/iRODS/Background/"                   # storage path of the baseline lib
    all_files : np.ndarray = np.asarray(os.listdir(baseline_dir))                   # container for all baseline files
    all_n_files : int = len(all_files)                                              # number of available baseline files

    def __init__(self, index : int = None) -> None : 

        self.__current_files = 0                                                    # number of traces already raised

        if index is None:
            random_file = RandomTrace.all_files[np.random.randint(RandomTrace.all_n_files)]
            these_traces = np.loadtxt(RandomTrace.baseline_dir + random_file)
        else:
            these_traces = np.loadtxt(RandomTrace.baseline_dir + RandomTrace.all_files[index])

        these_traces = np.split(these_traces, len(these_traces) // 3)               # group random traces by pmt    
        self._these_traces = np.array([station[:,1:] for station in these_traces])  # add them to this dataclass

    # get random traces for a single stations
    def get(self) -> np.ndarray : 
        
        if self.__current_files == len(self._these_traces) - 1: self.__init__()     # reload buffer on overflow
        self.__current_files += 1                                                   # update pointer after loading

        return self._these_traces[self.__current_files]

# container for injected muons
class InjectedBackground():

    # TODO get simulations for three different PMTs
    background_dir : str = "/cr/data01/filip/background/single_pmt.dat"             # storage path of the background lib
    library : np.ndarray = np.loadtxt(background_dir)                               # contains injected particle signals
    shape : tuple = library.shape                                                   # (number, length) of particles in ^

    # get n injected particles
    def __new__(self, n : int, trace_length : int) -> tuple : 
        
        Injections = np.zeros((3, trace_length))
        n_particles, length = InjectedBackground.shape
        injections_start, injections_end = [], []

        for _ in range(n):

            injected_particle = InjectedBackground.library[np.random.randint(n_particles)]
            injection_start = np.random.randint(0, trace_length - length)
            injection_end = injection_start + length

            for i in range(3): Injections[i][injection_start : injection_end] += injected_particle

            injections_start.append(injection_start)
            injections_end.append(injection_end)

        return injections_start, injections_end, Injections