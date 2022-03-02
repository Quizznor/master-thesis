#!/usr/bin/python3

import typing
import sys, os
import numpy as np
import tensorflow as tf

# Baseline trace containing <trace_length> bins
# Default baseline: mean == 0 ADC; std == 1 ADC
class VEMTrace():

    def __init__(self, trace_length : int, mu : float = 0, std : float = 1) -> None :

        self._array = np.random.normal(mu, std, trace_length)

    # overlay signal trace at random position of baseline
    def __iadd__(self, signal : np.array) -> object :
        assert trace_length > len(signal), "SIGNAL DOES NOT FIT INTO BASELINE!\n"
        start = np.random.randint(- trace_length, -len(signal))
        self._array[start : start + len(signal)] += signal

        return self

    # default getter for baseline trace
    def get(self) -> np.array : 
        return self._array

class Dataset():

    def __init__(self, fixed_seed : bool = True, shuffle : bool = False):

        # make random numbers predictable if desired
        fixed_seed and self.set_random_number_generator_seed(0)

        self.fixed_seed = fixed_seed    # set RNG seed for reproducible results
        self.shuffle = shuffle          # preshuffle signal/background generators

        # set filesystem binaries
        self._working_directory = "/cr/users/filip/data/first_simulation/tensorflow/signal/"
        self._signal_files = os.listdir(self._working_directory)
        
        # generate the dataset
        self.__dataset = tf.data.Dataset.from_generator(self._generator, (tf.float32, tf.int32), ( (20000,), ()))

    # prevent entire dataset from being loaded into memory via generators
    def _generator(self) -> typing.Generator[tuple, None, None] :

        # TODO Provide a way of looking at progress in stdout

        #########################
        # background labels == 0
        # signal labes      == 1
        #########################

        # generate signal events first, then baselines
        # definitely works, but memory expensive shuffling
        if not self.shuffle:

            # around 200k station-level traces available
            for file in self._signal_files:
                events = np.loadtxt(self._working_directory + file)
                for signal in events:
                    Baseline = VEMTrace(trace_length)
                    Baseline += signal
                    yield Baseline.get(), 1

            # Exactly n_signal background datapoints
            for step in range(n_signal):
                yield VEMTrace(trace_length).get(), 0
        
        # generate already mixed set of signals and baselines
        # more memory friendly, but needs to be tested 
        elif self.shuffle:
            for file in self._signal_files:
                events = np.loadtxt(self._working_directory + file)
                for signal in events:

                    # decide whether baseline gets interlaced
                    choice = np.random.choice([0,1])

                    while not choice:
                        yield VEMTrace(trace_length).get(), 0
                        choice = np.random.choice([0,1])

                    Baseline = VEMTrace(trace_length)
                    Baseline += signal
                    yield Baseline.get(), 1

    # return dataset main attribute
    def get(self):
        return self.__dataset

    # seed RNG for reproducibility
    @staticmethod
    def set_random_number_generator_seed(seed : int = 0) -> None :
        np.random.seed(seed)

    # no idea exactly what this does, see shorturl.at/dhpyQ for more info
    def get_inputs(self, n_shuffle : int, n_batch : int) -> tuple :

        # (re)set batch and shuffle size, just to be sure
        global shuffle_size, batch_size
        shuffle_size = n_shuffle
        batch_size   = n_batch
        
        # shuffle and group the dataset if needed
        if not self.shuffle:
            events = events.shuffle(shuffle_size)

        events = events.batch(batch_size)
        events = events.prefetch(1)

        features, labels = events.make_one_shot_iterator().get_next()
        return features, labels

    # takes roughly YES amount of time and RAM (=
    def show_data(self, n_shuffle : int, n_batch : int) -> None:

        features, labels = self.get_inputs(n_shuffle, n_batch)

        with tf.Session() as session:
            f_data, l_data = session.run([features, labels])
            for trace, label in zip(f_data, l_data):

                print("Preparing file...")

                color = "r" if label == 1 else "b"
                layer = 0 if label == 1 else 1
                plt.rcParams.update({'font.size': 22})
                plt.ylabel("Signal strength (VEM)")
                plt.xlabel("Time bin (8.3 ns)")

                plt.plot(range(len(trace)), trace, c = color, zorder = layer)

                plt.plot([],[], c = "r", label="Signal traces")
                plt.plot([],[], c = "b", label="Background traces")
                plt.legend()

                plt.savefig("/cr/users/filip/plots/first_model_test.png")

if __name__=="__main__":

    import matplotlib.pyplot as plt

    # Have 206563 station-level traces in dataset from first_simulation, see <check_signal_size.py> 

    n_signal = 206563               # total number of station-level traces in dataset
    shuffle_size = 2 * n_signal     # buffer size for shuffling, should be O(2 * n_signal)
    trace_length = 20000            # total trace "duration", 8.3 ns/bin * 20 000 bins = 166 μs
    batch_size = 1000               # number of points in batch for (e.g) gradient descent

    # initialize dataset (this doesn't load it into memory yet!)
    VirtualEventDataset = Dataset(shuffled=True)
    VirtualEventDataset.show_data()


    # initialize CNN model
    # TODO

