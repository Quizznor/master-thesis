import tensorflow as tf
import numpy as np
import typing

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
            pmt_data = trace_data[:,4:]

            # assert that metadata looks the same for all three PMTs
            for metadata in [station_ids, sp_distances, energies, zeniths]:
                if len(metadata) != 1:
                    raise ValueError

            self.StationID = next(iter(station_ids))
            self.SPDistance = next(iter(sp_distances))
            self.Energy = next(iter(energies))
            self.Zenith = next(iter(zeniths))

            # superimpose signal data on top of baseline at a random position
            if self.baseline_length < pmt_data.shape[1]:
                raise ValueError

            signal_start = np.random.randint(-self.trace_length, -pmt_data.shape[1])

            self.pmt_1[signal_start : signal_start + pmt_data.shape[1]] += pmt_data[0]
            self.pmt_2[signal_start : signal_start + pmt_data.shape[1]] += pmt_data[1]
            self.pmt_3[signal_start : signal_start + pmt_data.shape[1]] += pmt_data[2]

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