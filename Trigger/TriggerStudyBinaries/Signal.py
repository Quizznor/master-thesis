from dataclasses import dataclass
import numpy as np
import linecache
import typing
import sys

# library of stray (e.g.) muon signals that are accidentally injected
@dataclass
class Background():

    path    : str = "/cr/data01/filip/background/single_pmt.dat"                # storage path of the background lib
    library : np.ndarray = np.loadtxt(path)                                     # contains injected particle signals
    shape   : tuple = library.shape                                             # (number, length) of particles in ^

# Event wrapper for measurements of a SINGLE tank with 3 PMTs
class VEMTrace():

    # Common data for all VEM traces; Some of this can be overwritten in __init__
    background_frequency = 4665                                                 # frequency of accidental injections
    single_bin_duration = 8.3e-9                                                # time length of a single bin, in s                                               
    ADC_to_VEM_factor = 215.9                                                   # from David's Mail @ 7.06.22 3:30 pm
    trace_length = 2048                                                         # 1 Bin = 8.3 ns, 2048 Bins = ~17. µs
    baseline_std = 2                                                            # two FAD counts, NOT converted here!
    baseline_mean = [-2, 2]                                                     # same goes for (the limits) of means

    # metadata regarding shower origin, energy and so on
    StationID  = -1                                                             # set all these values to nonsensical
    Energy     = -1                                                             # numbers in the beginning for easier 
    SPDistance = -1                                                             # distinguishing between actual event 
    Zenith     = -1                                                             # and background traces down the line
    # TODO add timing information here? 
    # Might be needed for CDAS triggers ...

    def __init__(self, trace_data : np.ndarray = None, **kwargs) -> typing.NoReturn :

        r'''
        :trace_data ``tuple``: tuple with individual pmt data in each entry of the tuple. If None, background trace is raised

        :Keyword arguments:
            * *ADC_to_VEM* (``float``) -- ADC to VEM conversion factor, important for ub <-> uub
            * *n_bins* (``int``) -- generate a baseline with <trace_length> bins
            * *force_inject* (``int``) -- force the injection of <force_inject> pbackground particles
            * *sigma* (``float``) -- baseline std in ADC counts
            * *mu* (``list``) -- mean ADC level limit [low, high] in ADC counts

        Initialization fails if metadata doesn't match for the different PMTs
        or the baseline length is too short. In both cases a ValueError is raised 
        '''

        # Change VEM trace defaults (if desired)
        self.ADC_to_VEM_factor = self.set_trace_attribute(kwargs, "ADC_to_VEM", VEMTrace.ADC_to_VEM_factor)
        self.trace_length = self.set_trace_attribute(kwargs, "n_bins", VEMTrace.trace_length)
        self.baseline_std = self.set_trace_attribute(kwargs, "sigma", VEMTrace.baseline_std)
        self.baseline_mean = self.set_trace_attribute(kwargs, "mu", np.random.uniform(*VEMTrace.baseline_mean))

        # Create a Gaussian background for each PMT
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
            trace_data = self.convert_to_VEM(trace_data[:,4:])

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
        self.Injected = None

        n_inject = np.random.poisson( self.background_frequency * self.single_bin_duration * self.trace_length )
        self.n_injected = self.set_trace_attribute(kwargs, "force_inject", n_inject )

        if self.n_injected != 0:
            self.Injected = np.zeros((3, self.trace_length))
            self._bkg_injected_at, self._bkg_stopped_at = [], []

            for i in range(self.n_injected):
                particle = np.random.randint(0, Background.shape[0])
                background_particle = Background.library[particle]

                injected_at = np.random.randint(-self.trace_length, -Background.shape[1])
                self._bkg_stopped_at.append( injected_at + len(background_particle) + self.trace_length )
                self._bkg_injected_at.append( injected_at + self.trace_length )

                for j in range(3):
                    self.Injected[j][injected_at : injected_at + Background.shape[1]] += background_particle

        # Add all different components (Baseline, Injected, Signal) together
        self.pmt_1, self.pmt_2, self.pmt_3 = self.Baseline
        for Component in [self.Signal, self.Injected]:
            if Component is not None:
                self.pmt_1 += Component[0]
                self.pmt_2 += Component[1]
                self.pmt_3 += Component[2]


    # return a label and sector of the whole trace, specify window length and starting position
    def get_trace_window(self, start_bin : int, window_length : int) -> tuple :

        assert start_bin + window_length <= self.trace_length, "trace sector exceeds the whole trace length"

        start_at, stop_at = start_bin, start_bin + window_length
        cut = lambda array : array[start_at : stop_at]
        label = 0

        # check whether signal is in the given window frame
        try:
            for index in range(start_at, stop_at):
                if self._sig_injected_at <= index <= self._sig_stopped_at:
                    label = 1; break

        except AttributeError:
            label = 0

        return label, np.array([cut(self.pmt_1), cut(self.pmt_2), cut(self.pmt_3)])

    # return the number of bins containing a (signal, background) of a given window
    def get_n_signal_background_bins(self, index : int, window_length : int) -> tuple : 

        n_bkg_bins, n_sig_bins = 0, 0

        try:
            for start, stop in zip(self._bkg_injected_at, self._bkg_stopped_at):
                n_bkg_bins += self.count_bins(index, window_length, start, stop)
        except AttributeError:
            pass

        n_sig_bins = self.count_bins(index, window_length, self._sig_injected_at, self._sig_stopped_at)

        return n_sig_bins, n_bkg_bins

    # calculate the number of bins with overlayed signal in a given window
    @staticmethod
    def count_bins(index : int, window_length : int, start : int, stop : int) -> int : 

        bin_i, bin_f = index, index + window_length

        if bin_i > stop: return 0                                               # window is right of signal
        elif bin_f < start: return 0                                            # window is left of signal
        else: return min(bin_f, stop) - max(bin_i, start)                       # signal is contained in some form


    # return the trace either as a class with full information (reduce = False) or pooled / non-pooled
    def __call__(self, pooling : bool = True, reduce : bool = True) -> np.ndarray :

        if reduce: 
            # TODO: test more types of pooling here?
            if pooling:
                return np.array([max([self.pmt_1[i], self.pmt_2[i], self.pmt_3[i]]) for i in range(self.trace_length)])
            else:
                return np.array([self.pmt_1, self.pmt_2, self.pmt_3])
        else:
            return self

    # helper function for easier handling of kwargs upon initialization
    def set_trace_attribute(self, dict, key, fallback) -> typing.NoReturn:
        try:
            # baseline std and mean have to be converted to VEM first
            # baseline mean must be random (uniform) float between limits
            if key == "mu":
                    return np.random.uniform(*dict[key]) / self.ADC_to_VEM_factor
            elif key == "sigma":
                return dict[key] / self.ADC_to_VEM_factor
            elif key == "force_inject" and dict[key] == -1:
                return fallback
            else:
                return dict[key]
        except KeyError:
            return fallback

    # convert array of FADC counts to array of VEM counts
    def convert_to_VEM(self, signal : np.ndarray) -> np.ndarray :

        signal_VEM = []

        for pmt in signal:
            np.floor(pmt)
            signal_VEM.append(np.floor(pmt) / self.ADC_to_VEM_factor)

        return np.array(signal_VEM)

    # return a specific component of the trace (signal, injected, background)
    def get_component(self, key : str) -> np.ndarray :

        components = {"signal" : self.Signal,
                      "injected" : self.Injected,
                      "baseline" : self.Baseline}

        if components[key] is not None:
            return components[key]
        else:
            raise AttributeError(f"VEM trace does not have a component: {key}")

    # return the sum of individual vem trace bins
    def integrate(self) -> float : 
        return np.mean(np.sum([self.pmt_1, self.pmt_2, self.pmt_3], axis = 1))

    # plot the trace in whatever figure plt.gca() points to
    def plot(self, accumulate : bool = True) -> typing.NoReturn :

        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size': 22})

        plt.plot(range(self.trace_length), self.pmt_1, label = "PMT #1")
        plt.plot(range(self.trace_length), self.pmt_2, label = "PMT #2")
        plt.plot(range(self.trace_length), self.pmt_3, label = "PMT #3")

        plt.axvline(self.__sig_injected_at, ls = "--", c = "g")
        plt.axvline(self.__sig_stopped_at, ls = "--", c = "r")

        # plt.xlim(0,self.trace_length)
        plt.xlabel("Time bin / 8.3 ns")
        plt.ylabel("Signal / VEM")
        plt.legend()

        accumulate and plt.show()