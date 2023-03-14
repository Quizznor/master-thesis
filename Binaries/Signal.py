import random
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
        n_muons = set(pmt_data[:,4])
        n_electrons = set(pmt_data[:,5])
        n_photons = set(pmt_data[:,6])
        pmt_data = pmt_data[:,7:]

        assert trace_length > len(pmt_data[0]), "signal size exceeds trace length"

        # assert that metadata looks the same for all three PMTs
        for metadata in [station_ids, sp_distances, energies, zeniths]:
            assert len(metadata) == 1, "Metadata between PMTs doesn't match"

        self.StationID = int(next(iter(station_ids)))                               # the ID of the station in question
        self.SPDistance = int(next(iter(sp_distances)))                             # the distance from the shower core
        self.Energy = next(iter(energies))                                          # energy of the shower of this signal
        self.Zenith = next(iter(zeniths))                                           # zenith of the shower of this signal

        self.n_muons = int(next(iter(n_muons)))                                     # number of muons injected in trace
        self.n_electrons = int(next(iter(n_electrons)))                             # number of electrons injected in trace
        self.n_photons = int(next(iter(n_photons)))                                 # number of photons injected in trace

        self.particles = \
        {
            "mu" : self.n_muons,
            "muon" : self.n_muons,
            "muons" : self.n_muons,
            "e" : self.n_electrons,
            "electron" : self.n_electrons,
            "electrons" : self.n_electrons,
            "ph" : self.n_photons,
            "photon" : self.n_photons,
            "photons" : self.n_photons
        }

        self.Signal = np.zeros((3, trace_length))
        # self.signal_start = np.random.randint(0, trace_length - len(pmt_data[0]))
        self.signal_start = 660                                                     # use same latch bin as offline
        self.signal_end = self.signal_start + len(pmt_data[0])

        for i, PMT in enumerate(pmt_data):
            self.Signal[i][self.signal_start : self.signal_end] += PMT              # add signal onto mask


# container for the combined trace
class Trace(Signal):

    def __init__(self, baseline_data : np.ndarray, signal_data : tuple, trace_options : dict) :

        self.window_length = trace_options["window_length"]
        self.window_step = trace_options["window_step"]
        self.downsampled = trace_options["apply_downsampling"]
        self.trace_length = trace_options["trace_length"]
        self.simulation_q_charge = trace_options["simulation_q_charge"]
        self.simulation_q_peak = trace_options["simulation_q_peak"]
        self.baseline_q_charge = trace_options["baseline_q_charge"]
        self.baseline_q_peak = trace_options["baseline_q_peak"]
        self.injected = trace_options["force_inject"]

        if self.injected:
            self.injections_start, self.injections_end, self.Injected = InjectedBackground(self.injected, self.trace_length)
            self.has_accidentals = True
        else: 
            self.Injected = None
            self.has_accidentals = False

        # build Signal component
        if signal_data is not None: 
            super().__init__(signal_data, self.trace_length)
            self.has_signal = True
        else:
            self.Signal = None
            self._iteration_index = 0
            self.has_signal = False

        # build Baseline component
        self.Baseline = baseline_data

        # whether or not to apply downsampling
        if self.downsampled:
            self.random_phase = np.random.randint(0, 3)
            self._iteration_index = max(self.signal_start - self.window_length + np.random.randint(1, 10), 0) // 3 if self.has_signal else 0
        else: 
            self._iteration_index = max(self.signal_start - self.window_length + np.random.randint(1, 10), 0) if self.has_signal else 0

        # build the VEM trace and integral
        self.build_integral_trace()
        self.convert_to_VEM()
        
    # extract pmt data for a given trace window
    def get_trace_window(self, start_bin : int) -> np.ndarray : 

        stop_bin = start_bin + self.window_length
        pmt_1, pmt_2, pmt_3 = self.pmt_1[start_bin : stop_bin], self.pmt_2[start_bin : stop_bin], self.pmt_3[start_bin : stop_bin]
        assert len(pmt_1) == len(pmt_2) == len(pmt_3) == self.window_length, f"Invalid trace window [{len(pmt_1)}, {len(pmt_2)}, {len(pmt_3)}] at index {start_bin}; full shape: [{len(self.pmt_1)}, {len(self.pmt_2)}, {len(self.pmt_3)}]"

        return np.array([pmt_1, pmt_2, pmt_3])

    # convert from ADC counts to VEM_charge
    def build_integral_trace(self) -> None :

        # Signal and Injected are simulated, and have a constant/equal q_peak/q_charge
        # Baseline (for random traces) has different q_peak/q_charge and needs conversion
        conversion_factor = self.simulation_q_charge / self.baseline_q_charge

        # Technically, there should be a uniform distribution [0, 1] added here, to simulate ADC overflow
        # But...! Needs to be the exact same increment as in self.convert_to_VEM(), is this even relevant?
        baseline = np.array([pmt * conversion_factor[i] for i, pmt in enumerate(self.Baseline)])
        
        if self.has_signal: 
            baseline += self.Signal

        if self.has_accidentals:
            baseline += self.Injected

        # average across all PMTs to build integral trace
        # mathematically equivalent to averaging later
        # integral trace is always calculated from full bandwith trace

        self.int_1 = np.floor(baseline[0]) / self.simulation_q_charge[0]
        self.int_2 = np.floor(baseline[1]) / self.simulation_q_charge[1]
        self.int_3 = np.floor(baseline[2]) / self.simulation_q_charge[2]
        self.deposited_signal = [np.sum(self.int_1), np.sum(self.int_2), np.sum(self.int_3)]
        self.integral_window = np.empty(self.trace_length)
        self.integral_window.fill(np.NaN)

        for i in range(self._iteration_index, self.trace_length, self.window_step):
            start_bin = i if not self.downsampled else i * 3
            stop_bin = i + self.window_length if not self.downsampled else (i + self.window_length) * 3
            w1, w2, w3 = self.int_1[start_bin : stop_bin], self.int_2[start_bin : stop_bin], self.int_3[start_bin : stop_bin]
            self.integral_window[i] = np.mean([sum(w1), sum(w2), sum(w3)])

    # convert from ADC counts to VEM_peak 
    def convert_to_VEM(self) -> None :

        # Signal + Injections ALWAYS have simulated q_peak/q_area 
        # Background has simulated q_peak/q_area if NOT random traces
        # otherwise set to values defined in RandomTrace class (l281)

        # convert Baseline from "real" q_peak/charge to simulated
        # simulate a floating point baseline for realistic bin overflow
        conversion_factor = self.simulation_q_peak / self.baseline_q_peak
        self.Baseline += np.random.uniform(size = self.Baseline.shape)
        self.Baseline = np.array([pmt * c for c, pmt in zip(conversion_factor, self.Baseline)])

        self.pmt_1, self.pmt_2, self.pmt_3 = np.zeros((3, 2048) )

        for component in [self.Baseline, self.Injected, self.Signal]:
            if component is None: continue

            self.pmt_1 += component[0]
            self.pmt_2 += component[1]
            self.pmt_3 += component[2]

        if self.downsampled:
            self.pmt_1 = np.floor(self.apply_downsampling(self.pmt_1, self.random_phase)) / self.simulation_q_peak[0]
            self.pmt_2 = np.floor(self.apply_downsampling(self.pmt_2, self.random_phase)) / self.simulation_q_peak[1]
            self.pmt_3 = np.floor(self.apply_downsampling(self.pmt_3, self.random_phase)) / self.simulation_q_peak[2]

            if self.has_signal:
                self.signal_start = int(self.signal_start / 3)
                self.signal_end = int(self.signal_end / 3)
            
            if self.has_accidentals:
                self.injections_start = [int(start / 3) for start in self.injections_start ]
                self.injections_end = [int(end / 3) for end in self.injections_end ]

            self.trace_length = self.trace_length // 3
        else:
            self.pmt_1 = np.floor(self.pmt_1) / self.simulation_q_peak[0]
            self.pmt_2 = np.floor(self.pmt_2) / self.simulation_q_peak[1]
            self.pmt_3 = np.floor(self.pmt_3) / self.simulation_q_peak[2]

        self.Baseline = np.floor(self.Baseline)

    @staticmethod
    def apply_downsampling(pmt, random_phase) -> np.ndarray :

        # ensure downsampling works as intended
        # cuts away (at most) the last two bins
        if len(pmt) % 3 != 0: pmt = pmt[0 : -(len(pmt) % 3)]

        # see /cr/data01/filip/offline/trunk/Framework/SDetector/UUBDownsampleFilter.h for more information
        kFirCoefficients = [ 5, 0, 12, 22, 0, -61, -96, 0, 256, 551, 681, 551, 256, 0, -96, -61, 0, 22, 12, 0, 5 ]
        buffer_length = int(0.5 * len(kFirCoefficients))
        kFirNormalizationBitShift = 11
        # kADCsaturation = 4095                             # bit shift not really needed

        n_bins_uub      = (len(pmt) // 3) * 3               # original trace length
        n_bins_ub       = n_bins_uub // 3                   # downsampled trace length
        sampled_trace   = np.zeros(n_bins_ub)               # downsampled trace container

        temp = np.zeros(n_bins_uub + len(kFirCoefficients))

        temp[0 : buffer_length] = pmt[:: -1][-buffer_length - 1 : -1]
        temp[-buffer_length - 1: -1] = pmt[:: -1][0 : buffer_length]
        temp[buffer_length : -buffer_length - 1] = pmt

        # perform downsampling
        for j, coeff in enumerate(kFirCoefficients):
            sampled_trace += [temp[k + j] * coeff for k in range(random_phase, n_bins_uub, 3)]

        # clipping and bitshifting
        sampled_trace = [int(adc) >> kFirNormalizationBitShift for adc in sampled_trace]

        # # clipping and bitshifting
        # for j, adc in enumerate(sampled_trace):
        #     # sampled_trace[i,j] = np.clip(int(adc) >> kFirNormalizationBitShift, a_min = -20, a_max = None)              # why clip necessary, why huge negative values?
        #     sampled_trace[j] = int(adc) >> kFirNormalizationBitShift

        return np.array(sampled_trace)

    # make this class an iterable
    def __iter__(self) -> typing.Union[tuple, StopIteration] : 

        while True:
            
            # only iterate over Signal region
            if self.has_signal:
                if self._iteration_index > self.signal_end: return StopIteration
            
            # iterate over everything in Background trace
            if self._iteration_index + self.window_length > self.trace_length: return StopIteration
            
            yield self.get_trace_window(self._iteration_index)
            self._iteration_index += self.window_step

    # wrapper for pretty printing
    def __repr__(self) -> str :

        reduce_by = 10 if self.downsampled else 30
        trace = list(" " * (self.trace_length // reduce_by))

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

        else: metadata = " Background trace                "
        
        return "||" + "".join(trace) + "||" + metadata

    # wrapper for plotting a trace
    def __plot__(self) -> None :

        x = range(self.trace_length)
        
        try:
            plt.title(f"Station #{self.StationID} - {np.mean(self.deposited_signal):.2f} VEM")
        except AttributeError:
            plt.title(f"Background trace - {self.deposited_signal:.2f} VEM")
            
        plt.plot(x, self.pmt_1, c = "steelblue", label = f"PMT #1{' - downsampled' if self.downsampled else ''}, S = {self.deposited_signal[0] :.1f} VEM", lw = 1)
        plt.plot(x, self.pmt_2, c = "orange", label = f"PMT #2{' - downsampled' if self.downsampled else ''}, S = {self.deposited_signal[1]:.1f} VEM", lw = 1)
        plt.plot(x, self.pmt_3, c = "green", label = f"PMT #3{' - downsampled' if self.downsampled else ''}, S = {self.deposited_signal[2]:.1f} VEM", lw = 1)


        if self.has_signal:
            plt.axvline(self.signal_start, ls = "--", c = "red", lw = 2)
            plt.axvline(self.signal_end, ls = "--", c = "red", lw = 2)

        if self.has_accidentals:
            for start, stop in zip(self.injections_start, self.injections_end):
                plt.axvline(start, ls = "--", c = "gray")
                plt.axvline(stop, ls = "--", c = "gray")

        plt.xlim(0, self.trace_length)
        plt.ylabel("Signal strength / VEM")
        plt.xlabel("Bin / 25 ns" if self.downsampled else "Bin / 8.3 ns")
        plt.legend()
        plt.show()


# container for reading signal files
class SignalBatch():

    def __new__(self, trace_file : str) -> np.ndarray :

        # print(f"\n[INFO] -- READING {'/'.join(trace_file.split('/')[-3:])}" + 20 * " ", end = "\r")

        with open(trace_file, "r") as file:
                signal = [[float(x) for x in line.split()] for line in file.readlines()]

        return [np.array([signal[i], signal[i + 1], signal[i + 2]]) for i in range(0, len(signal), 3)]


# container for gaussian baseline
class Baseline():

    def __new__(self, mu : float, sigma : float, length : int) -> tuple[float, float, np.ndarray] :
        return np.random.normal(mu, sigma, (3, length))


# container for random traces
class RandomTrace():

    baseline_dir : str = "/cr/tempdata01/filip/iRODS/"                              # storage path of the station folders
    # all_files : np.ndarray = np.asarray(os.listdir(baseline_dir))                   # container for all baseline files
    # all_n_files : int = len(all_files)                                              # number of available baseline files

    def __init__(self, station : str = None, index : int = None) -> None : 

        ## (HOPEFULLY) TEMPORARILY FIXED TO NURIA/LO_QUI_DON DUE TO BAD FLUCTUATIONS IN OTHER STATIONS
        # self.station = random.choice(["nuria", "lo_qui_don"]) if station is None else station.lower()
        self.station = "nuria" if station is None else station
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

        print(f"\n[INFO] -- LOADING {self.station.upper()}: {self.random_file}" + 20 * " ", end = "\n")

        these_traces = np.loadtxt(RandomTrace.baseline_dir + self.station + "/" + self.random_file)

        # IF YOU WANT TO USE DAY AVERAGE FROM ONLINE ESTIMATE #########################################
        # values come from $TMPDATA/iRODS/MonitoringData/read_monitoring_data.ipynb -> monitoring files
        # scale factors for correct trigger rates are at ~/Trigger/RunProductionTest/plot_everything.ipynb

        if "nuria" in self.station:
            self.q_peak = np.array([180.23, 182.52, 169.56]) * (1 - 11.59/100)
            self.q_charge = np.array([3380.59, 3508.69, 3158.88])      # scaling factor ???
        elif "lo_qui_don" in self.random_file:
            self.q_peak = np.array([163.79, 162.49, 173.71]) * (1 - 10.99/100)
            self.q_charge = np.array([2846.67, 2809.48, 2979.65])      # scaling factor ???
        elif "jaco" in self.random_file:
            self.q_peak = np.array([189.56, 156.48, 168.20])
            self.q_charge = np.array([3162.34, 2641.25, 2840.97])
        elif "peru" in self.random_file:
            self.q_peak = np.array([164.02, 176.88, 167.37])
            self.q_charge = np.array([2761.37, 3007.72, 2734.63])
        else:
            print("[WARN] -- Station not found! Using OFFLINE estimate")
            self.q_peak = [GLOBAL.q_peak for i in range(3)]
            self.q_charge = [GLOBAL.q_charge for i in range(3)]

        self._these_traces = np.split(these_traces, len(these_traces) // 3)         # group random traces by pmt


    # get random traces for a single stations
    def get(self) -> tuple[float, float, np.ndarray] : 
        
        try:                                                                        # update pointer after loading
            self.__current_traces += 1

            return self.q_peak, self.q_charge, self._these_traces[self.__current_traces]
        
        except IndexError:                                                          # reload buffer on overflow

            try:
                self.__init__(station = self.station, index = self.index + 1)
            except TypeError:
                self.__init__(station = self.station)

            return self.get()


# container for injected muons
class InjectedBackground():

    # TODO get simulations for three different PMTs
    background_dir : str = "/cr/data01/filip/background/single_pmt.dat"             # storage path of the background lib
    library : np.ndarray = np.loadtxt(background_dir)                               # contains injected particle signals
    shape : tuple = library.shape                                                   # (number, length) of particles in ^

    # get n injected particles
    def __new__(self, n : int, trace_length : int) -> tuple :

        if isinstance(n, str): n = self.poisson()
        
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
    
    # poissonian for background injection
    @staticmethod
    def poisson() -> int :
        return np.random.poisson( GLOBAL.background_frequency * GLOBAL.single_bin_duration * GLOBAL.trace_length )