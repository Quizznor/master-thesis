import numpy as np
import typing
import sys

# Event wrapper for measurements of a SINGLE tank with 3 PMTs
class VEMTrace():

    is_background = True                                                        # flag for background vem traces

    # defaults for the VEM traces; Can be overwritten in __init__
    ADC_to_VEM_factor = 215.9                                                   # from David's Mail @ 7.06.22 3:30 pm
    baseline_length = 20000                                                     # 1 Bin = 8.3 ns, 20000 Bins = 166 us
    baseline_std = 0.5 / ADC_to_VEM_factor                                      # half an adc, converted to VEM units
    baseline_mean = [-0.5 / ADC_to_VEM_factor, +0.5 / ADC_to_VEM_factor]        # same goes for (the limits) of means

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
            * *ADC_to_VEM* (``float``) -- ADC to VEM conversion factor, for ub <-> uub (i think), for example
            * *n_bins* (``int``) -- generate a baseline with <trace_length> bins
            * *sigma* (``float``) -- baseline std in ADC counts (gets converted automatically)
            * *mu* (``list``) -- mean ADC level limit [low, high] in ADC counts

        Initialization fails if metadata doesn't match for the different PMTs
        or the baseline length is too short. In both cases a ValueError is raised 
        '''

        # Change VEM trace defaults (if desired)
        self.ADC_to_VEM_factor = self.set_trace_attribute(kwargs, "ADC_to_VEM", VEMTrace.ADC_to_VEM_factor)
        self.trace_length = self.set_trace_attribute(kwargs, "n_bins", VEMTrace.baseline_length)
        self.baseline_std = self.set_trace_attribute(kwargs, "sigma", VEMTrace.baseline_std)
        self.baseline_mean = self.set_trace_attribute(kwargs, "mu", np.random.uniform(*VEMTrace.baseline_mean))

        # Create a baseline for each PMT
        self.pmt_1 = np.random.normal(self.baseline_mean, self.baseline_std, self.trace_length)
        self.pmt_2 = np.random.normal(self.baseline_mean, self.baseline_std, self.trace_length)
        self.pmt_3 = np.random.normal(self.baseline_mean, self.baseline_std, self.trace_length)

        # Accidentally inject background particles
        # TODO

        # Add a signal on top of baseline
        if trace_data is not None:

            self.is_background = False

            # every line in the input files looks like this:
            # [ID SP_Distance E Theta bin_1 bin_2 ... bin_2048]
            
            # group trace information first
            station_ids = set(trace_data[:,0])
            sp_distances = set(trace_data[:,1])
            energies = set(trace_data[:,2])
            zeniths = set(trace_data[:,3])

            # assert that metadata looks the same for all three PMTs
            for metadata in [station_ids, sp_distances, energies, zeniths]:
                if len(metadata) != 1:
                    raise ValueError("Metadata between PMTs doesn't match")

            self.StationID = next(iter(station_ids))                            # the ID of the station in question
            self.SPDistance = next(iter(sp_distances))                          # the distance from the shower core
            self.Energy = next(iter(energies))                                  # energy of the shower of this signal
            self.Zenith = next(iter(zeniths))                                   # zenith of the shower of this signal

            # TODO: Is this order of operations the right way?
            self.Signal = self.convert_to_VEM(np.array(trace_data[:,4:]))       # container the data measured by the pmts

            # superimpose signal data on top of baseline at a random position
            if self.baseline_length < self.Signal.shape[1]:
                raise ValueError("Signal does not fit into baseline!")

            signal_start = np.random.randint(-self.trace_length, -self.Signal.shape[1])

            self.pmt_1[signal_start : signal_start + self.Signal.shape[1]] += self.Signal[0]
            self.pmt_2[signal_start : signal_start + self.Signal.shape[1]] += self.Signal[1]
            self.pmt_3[signal_start : signal_start + self.Signal.shape[1]] += self.Signal[2]

    # getter for easier handling of PMT data
    def __call__(self, pooling : bool = True, reduce : bool = True) -> np.ndarray :

        if reduce:
            if pooling:
                return np.array([max([self.pmt_1[i], self.pmt_2[i], self.pmt_3[i]]) for i in range(self.baseline_length)])
            else:
                return np.array([self.pmt_1, self.pmt_2, self.pmt_3])
        else:
            return self

    # helper function for easier handling of kwargs upon initialization
    def set_trace_attribute(self, dict, key, fallback):
        try:
            # baseline std and mean have to be converted to VEM first
            # baseline mean must be random (uniform) float between limits
            if key == "mu":
                    return np.random.uniform(*dict[key]) / self.ADC_to_VEM_factor
            elif key == "sigma":
                return dict[key] / self.ADC_to_VEM_factor
            else:
                return dict[key]
        except KeyError:
            return fallback

    def convert_to_VEM(self, signal : np.ndarray) -> np.ndarray :

        signal_VEM = []

        for pmt in signal:
            signal_VEM.append(np.floor(pmt) / self.ADC_to_VEM_factor)

        return np.array(signal_VEM)

    def get_signal(self) -> tuple : 
        if self.signal is not None:
            return self.signal
        else:
            raise AttributeError("VEM trace does not have a signal")

    def integrate(self) -> float : 
        return np.mean(np.sum([self.pmt_1, self.pmt_2, self.pmt_3], axis = 1))

    def plot(self) -> typing.NoReturn :

        if "matplotlib.pyplot" not in sys.modules:
            import matplotlib.pyplot as plt
            plt.rcParams.update({'font.size': 22})

        ax1 = plt.subplot2grid((3,1),(0,0))
        ax2, ax3 = [plt.subplot2grid((3,1),(i,0), sharex = ax1, sharey = ax1) for i in range(1,3)]

        ax1.plot(range(self.trace_length), self.pmt_1)
        ax2.plot(range(self.trace_length), self.pmt_3)
        ax3.plot(range(self.trace_length), self.pmt_3)

        ax1.set_xlim(0,self.trace_length)
        ax1.set_xlabel("Time bin / 8.3 ns")
        ax1.set_ylabel("PMT #1 / VEM")
        ax2.set_ylabel("PMT #2 / VEM")
        ax3.set_ylabel("PMT #3 / VEM")

        plt.show()