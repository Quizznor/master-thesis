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
            self.Signal[i][self.signal_start : self.signal_end] += PMT              # add signal onto mask


# container for the combined trace
class Trace(Signal):

    def __init__(self, trace_options : list, baseline_data : np.ndarray, signal_data : tuple = None) :

        self.q_peak = trace_options[0]
        self.q_charge = trace_options[1]
        self.length = trace_options[2]
        self.sigma = trace_options[3]
        self.mu = trace_options[4]

        if trace_options[5] is not None:
            self.inject = trace_options[5]
        else: self.inject = self.poisson()

        self.downsample = trace_options[6]

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
            self.pmt_1, self.pmt_2, self.pmt_3 = self.convert_to_VEM( self.Baseline + self.Signal + self.Injected, self.q_peak )
        elif self.has_accidentals and not self.has_signal:
            self.pmt_1, self.pmt_2, self.pmt_3 = self.convert_to_VEM( self.Baseline + self.Injected, self.q_peak )
        elif self.has_signal and not self.has_accidentals:
            self.pmt_1, self.pmt_2, self.pmt_3 = self.convert_to_VEM( self.Baseline + self.Signal, self.q_peak )
        else: self.pmt_1, self.pmt_2, self.pmt_3 = self.convert_to_VEM( self.Baseline, self.q_peak )

    # poissonian for background injection
    def poisson(self) -> int :

        return np.random.poisson( GLOBAL.background_frequency * GLOBAL.single_bin_duration * self.length )

    # extract pmt data plus label for a given trace window 
    def get_trace_window(self, window : tuple) -> tuple : 

        start, stop = window

        pmt_1, pmt_2, pmt_3 = self.pmt_1[start : stop], self.pmt_2[start : stop], self.pmt_3[start : stop]
        label = self.calculate_signal_overlap(window)
        integral = self.integrate(window)

        # used some time ago for debugging purposes
        # if np.array([pmt_1, pmt_2, pmt_3]).shape != (3,120):
            
        #     print(window, self.signal_start, self.signal_end)
        #     raise StopIteration

        return np.array([pmt_1, pmt_2, pmt_3]), label, integral

    # calculate number of bins of signal in sliding window
    def calculate_signal_overlap(self, window : tuple) -> int :
        
        if self.has_signal: 
            return len(range(max(window[0], self.signal_start), min(window[-1], self.signal_end)))
        else: return 0

    # return the mean of integrated PMT signals (VEM_charge) for a given window
    def integrate(self, window : np.ndarray) -> float : 

        start, stop = window
        trace_window = self.Baseline[:, start : stop] / self.q_charge

        if self.has_accidentals: trace_window += self.Injected[:, start : stop] / GLOBAL.q_charge
        if self.has_signal: trace_window += self.Signal[:, start : stop] / GLOBAL.q_charge

        return np.mean(np.sum(trace_window, axis = 1))

    # convert from ADC counts to VEM 
    def convert_to_VEM(self, signal : np.ndarray, ADC_to_VEM : float) -> np.ndarray :

        if self.downsample: 
            signal = self.apply_downsampling(signal)

            if self.has_signal:
                self.signal_start = self.signal_start // 3
                self.signal_end = self.signal_end // 3

            if self.has_accidentals:
                self.injections_start = [start // 3 for start in self.injections_start ]
                self.injections_end = [end // 3 for end in self.injections_end ]

            self.length = self.length // 3

        return np.floor(signal) / ADC_to_VEM

    @staticmethod
    def apply_downsampling(trace : np.ndarray) -> np.ndarray :

        # ensure downsampling works as intended
        # cuts away (at most) the last two bins
        if trace.shape[-1] % 3 != 0:
            trace = np.array([pmt[0 : -(trace.shape[-1] % 3)] for pmt in trace])

        # see /cr/data01/filip/offline/trunk/Framework/SDetector/UUBDownsampleFilter.h for more information
        kFirCoefficients = [ 5, 0, 12, 22, 0, -61, -96, 0, 256, 551, 681, 551, 256, 0, -96, -61, 0, 22, 12, 0, 5 ]
        buffer_length = int(0.5 * len(kFirCoefficients))
        kFirNormalizationBitShift = 11
        # kADCSaturation = 4095                             # bit shift not really needed

        n_bins_uub      = np.shape(trace)[1]                # original trace length
        n_bins_ub       = int(n_bins_uub / 3)               # downsampled trace length
        sampled_trace   = np.zeros((3, n_bins_ub))          # the downsampled trace


        temp = np.zeros(n_bins_uub + len(kFirCoefficients))

        for i_pmt, pmt in enumerate(trace):

            temp[0 : buffer_length] = pmt[:: -1][-buffer_length - 1 : -1]
            temp[-buffer_length - 1: -1] = pmt[:: -1][0 : buffer_length]
            temp[buffer_length : -buffer_length - 1] = pmt

            # perform downsampling
            for j, coeff in enumerate(kFirCoefficients):
                sampled_trace[i_pmt] += [temp[k + j] * coeff for k in range(0, n_bins_uub, 3)]

        # clipping and bitshifting
        for i, pmt in enumerate(sampled_trace):
            for j, adc in enumerate(pmt):
                # sampled_trace[i,j] = np.clip(int(adc) >> kFirNormalizationBitShift, a_min = None, a_max = kADCSaturation)
                sampled_trace[i,j] = int(adc) >> kFirNormalizationBitShift

        return sampled_trace

    # wrapper for pretty printing
    def __repr__(self) -> str :

        reduce_by = 10 if self.downsample else 30
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

        plt.plot(x, self.pmt_1, c = "green", label = f"PMT #1{', downsampled' if self.downsample else ''}", lw = 1)
        plt.plot(x, self.pmt_2, c = "orange", label = f"PMT #2{', downsampled' if self.downsample else ''}", lw = 1)
        plt.plot(x, self.pmt_3, c = "steelblue", label = f"PMT #3{', downsampled' if self.downsample else ''}", lw = 1)

        if self.has_signal:
            plt.axvline(self.signal_start, ls = "--", c = "red", lw = 2)
            plt.axvline(self.signal_end, ls = "--", c = "red", lw = 2)

        if self.has_accidentals:
            for start, stop in zip(self.injections_start, self.injections_end):
                plt.axvline(start, ls = "--", c = "gray")
                plt.axvline(stop, ls = "--", c = "gray")

        plt.xlim(0, self.length)
        plt.ylabel("Signal strength / VEM")
        plt.xlabel("Bin / 25 ns" if self.downsample else "Bin / 8.3 ns")
        plt.legend()
        plt.show()


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
# TODO pair with q_peak and q_charge values
class RandomTrace():

    baseline_dir : str = "/cr/tempdata01/filip/iRODS/corrected/"                    # storage path of the baseline lib
    all_files : np.ndarray = np.asarray(os.listdir(baseline_dir))                   # container for all baseline files
    all_n_files : int = len(all_files)                                              # number of available baseline files

    def __init__(self, index : int = None) -> None : 

        self.__current_files = 0                                                    # number of traces already raised

        if index is None:
            random_file = RandomTrace.all_files[np.random.randint(RandomTrace.all_n_files)]
            these_traces = np.loadtxt(RandomTrace.baseline_dir + random_file)
        else:
            these_traces = np.loadtxt(RandomTrace.baseline_dir + RandomTrace.all_files[index])

        self._these_traces = np.split(these_traces, len(these_traces) // 3)         # group random traces by pmt

    # get random traces for a single stations
    def get(self) -> np.ndarray : 
        
        try:
            self.__current_files += 1                                               # update pointer after loading

            # hack for now, replace with real q_peak, q_charge down the line
            return GLOBAL.q_peak, GLOBAL.q_charge, self._these_traces[self.__current_files]
        
        except IndexError:                                                          # reload buffer on overflow

            self.__init__()
            return self.get()

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