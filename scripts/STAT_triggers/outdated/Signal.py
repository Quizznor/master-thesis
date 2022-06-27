import typing, sys, os
import numpy as np

# Event wrapper for measurements of a SINGLE tank with 3 PMTs
class VEMTrace():

    def __init__(self, label : str, *args, **kwargs) -> typing.NoReturn :

        r'''
        :label ``str``:
            one of "SIG" or "BKG" -- specifies the type of trace

        :Arguments (in order):
            * *trace_length* (``int``) -- generate a baseline with <trace_length> bins
            * *baseline_std* (``float``) -- baseline std in VEM counts
            * *baseline_mean* (``list``) -- mean ADC level limit [low, high]
            * *pooling* (``bool``) -- apply max pooling to 3 PMT tuple to reduce data size
            * *trace* (``list ``) -- the trace data of shape [PMT1, PMT2, PMT3]
        '''

        self.label = label      # one of "SIG" or "BKG"        

        # full initialization of trace on first call
        try:

            # set trace shape characteristics first
            self.trace_length = args[0]

            # set baseline std (default exactly 0.5 ADC)
            self.baseline_std = args[1]

            # Reduce input dimensionality by pooling
            self.pooling = args[3]

            # set baseline mean (default in [-0.5, 0.5] ADC)
            self.baseline_mean = np.random.uniform(*args[2])

            # create baseline VEM trace, same mean and std (TODO different std?)
            self._pmt_1 = np.random.normal(self.baseline_mean, self.baseline_std, self.trace_length)
            self._pmt_2 = np.random.normal(self.baseline_mean, self.baseline_std, self.trace_length)
            self._pmt_3 = np.random.normal(self.baseline_mean, self.baseline_std, self.trace_length)

            if label == "SIG": 

                def group_and_check_metadata(metadata : list) -> float :
                    assert len(set(metadata)) == 1, "SPDistance, Energy or Zenith don't match for PMTs"
                    return set(metadata).pop()

                # extract metadata from trace file (energy, zenith, spdistance)
                # 0th bin = SPDistance, 1st bin = energy, 2nd bin = zenith
                # all following bins = trace (should be of length 2048 = ~17 µs)
                self._StationID  = group_and_check_metadata(args[4][:,0])
                self._SPDistance = group_and_check_metadata(args[4][:,1])
                self._Energy     = group_and_check_metadata(args[4][:,2])
                self._Zenith     = group_and_check_metadata(args[4][:,3])

                # ectract the PMT signals from trace file
                signal_length = len(args[4][0]) - 4
                vem_signals = args[4][:,4:]

                assert len(vem_signals[0]) == len(vem_signals[1]) == len(vem_signals[2]), "SIGNAL SHAPES DONT MATCH!\n"
                assert self.trace_length > signal_length, "SIGNAL DOES NOT FIT INTO BASELINE!\n"

                # overlay signal shape at random position of baseline
                start = np.random.randint(-self.trace_length, -signal_length)
                self._pmt_1[start : start + signal_length] += vem_signals[0]
                self._pmt_2[start : start + signal_length] += vem_signals[1]
                self._pmt_3[start : start + signal_length] += vem_signals[2]

            elif label == "BKG":
                self._StationID  = -1
                self._SPDistance = -1
                self._Energy     = -1
                self._Zenith     = -1

        except IndexError:

            try:

                self._pmt_1, self._pmt_2, self._pmt_3 = np.split(kwargs['trace'], 3)
                # assert self._pmt_1.shape == self._pmt_2.shape == self._pmt_3.shape, "SIGNAL SHAPES DONT MATCH!\n"
                self.trace_length = len(self._pmt_1)

            except ValueError:
                self._pmt_1 = self._pmt_2 = self._pmt_3 = kwargs['trace']

    # getter for easier handling of data classes
    def __call__(self) -> tuple :

        # this is a mess, lmao
        try:
            if self.pooling:
                return np.array([max([self._pmt_1[i], self._pmt_2[i], self._pmt_3[i]]) for i in range(len(self._pmt_1))])
            elif not self.pooling:
                return np.array(list(self._pmt_1) + list(self._pmt_2) + list(self._pmt_3))
        except AttributeError:
            return np.array([max([self._pmt_1[i], self._pmt_2[i], self._pmt_3[i]]) for i in range(len(self._pmt_1))])

    # # plot the VEM signal
    # def plot(self) -> typing.NoReturn : 

    #     # TODO: more traces in one plot?

    #     fig, (ax1, ax2, ax3) = plt.subplots(nrows = 3, ncols = 1, sharex = True, sharey = True)
    #     ax1.set_title("PMT #1"), ax2.set_title("PMT #2"), ax3.set_title("PMT #3")
    #     ax2.set_ylabel("Signal strength (VEM)")
    #     ax3.set_xlabel("Time bin (8.3 ns)")

    #     # fig.subptitle(f"Event: E = {self._Energy:.2e} eV, $\Theta$ = {self._Zenith:.2f}°")
    #     ax1.plot(range(len(self._pmt_1)),self._pmt_1), ax2.plot(range(len(self._pmt_2)), self._pmt_2), ax3.plot(range(len(self._pmt_3)), self._pmt_3)
    #     # ax1.plot(ra)
