import typing, sys, os
import numpy as np

# Event wrapper for measurements of a SINGLE tank with 3 PMTs
class VEMTrace():

    def __init__(self, label : str, *args, **kwargs) -> typing.NoReturn :

        self.label = label      # one of "SIG" or "BKG"
        

        # full initialization of trace on first call
        try:

            # set trace shape characteristics first
            self.trace_length = args[0]

            # set baseline std (default exactly 0.5 ADC)
            self.baseline_std = args[1]

            # Reduce input dimensionality by pooling
            self.pooling = args[3]  # True or False

            # set baseline mean (default in [-0.5, 0.5] ADC)
            self.baseline_mean = np.random.uniform(*args[2])

            # create baseline VEM trace, same mean and std (TODO different std?)
            self.__pmt_1 = np.random.normal(self.baseline_mean, self.baseline_std, self.trace_length)
            self.__pmt_2 = np.random.normal(self.baseline_mean, self.baseline_std, self.trace_length)
            self.__pmt_3 = np.random.normal(self.baseline_mean, self.baseline_std, self.trace_length)

            if label == "SIG": 

                # don't catch exception here, since we _need_ signal data to continue
                signal_length = len(args[4][0])
                vem_signals = args[4]

                # assert len(vem_signals[0]) == len(vem_signals[1]) == len(vem_signals[2]), "SIGNAL SHAPES DONT MATCH!\n"
                # assert self.trace_length > signal_length, "SIGNAL DOES NOT FIT INTO BASELINE!\n"

                # overlay signal shape at random position of baseline
                start = np.random.randint(-self.trace_length, -signal_length)
                self.__pmt_1[start : start + signal_length] += vem_signals[0]
                self.__pmt_2[start : start + signal_length] += vem_signals[1]
                self.__pmt_3[start : start + signal_length] += vem_signals[2]

            elif label == "BKG":
                pass

        except IndexError:

            try:

                self.__pmt_1, self.__pmt_2, self.__pmt_3 = np.split(kwargs['trace'], 3)
                # assert self.__pmt_1.shape == self.__pmt_2.shape == self.__pmt_3.shape, "SIGNAL SHAPES DONT MATCH!\n"
                self.trace_length = len(self.__pmt_1)

            except ValueError:
                self.__pmt_1 = self.__pmt_2 = self.__pmt_3 = kwargs['trace']

    # getter for easier handling of data classes
    def __call__(self) -> tuple :

        if self.pooling:
            return [max([self.__pmt_1[i], self.__pmt_2[i], self.__pmt_3[i]]) for i in range(len(self.__pmt_1))]
        elif not self.pooling:
            return list(self.__pmt_1) + list(self.__pmt_2) + list(self.__pmt_3)
