#!/usr/bin/python3

import typing
import sys, os
import numpy as np
import tensorflow as tf

# Baseline trace containing <trace_length> bins
# Default baseline: mean == 0 ADC; std == 1 ADC
class VEMTrace():

    def __init__(self, trace_length : int, mu : float = 0, std : float = 1) -> None :

        self.trace_length = trace_length                            # number of time bins (8.3 ns) in trace
        self.__array = np.random.normal(mu, std, trace_length)      # gaussian baseline with mean and std
        
    # overlay signal trace at uniformly random position of baseline
    def __iadd__(self, signal : np.array) -> object :
        assert self.trace_length > len(signal), "SIGNAL DOES NOT FIT INTO BASELINE!\n"

        start = np.random.randint(-self.trace_length, -len(signal))
        self.__array[start : start + len(signal)] += signal

        return self

    # default getter for baseline trace
    def get(self) -> np.array : 
        return self.__array

# Generator class for NN sequential model with some additional functionalities
# This website helped tremendously with writing a working example: shorturl.at/fFI09
class TraceGenerator(tf.keras.utils.Sequence):

    def __init__(self, train : bool, split : float, input_shape : int, fix_seed : bool = False, shuffle : bool = True, verbose : bool = True ) -> None:

        assert 0 <= split <= 1, "PLEASE PROVIDE A VALID SPLIT: 0 < split < 1"

        self.train = train              # wether object is used for training or validation
        self.split = int(100 * split)   # 0 - 100, percentage of data for this generator
        self.input_shape = input_shape  # input length of the data, trace duration = 8.3 ns * length
        self.fix_seed = fix_seed        # whether or not RNG seed is fixed, for reproducibility
        self.shuffle = shuffle          # whether or not to shuffle signal files at the end of each epoch
        self.verbose = verbose          # whether or not to print output to stdout during training

        self.n_events = [0, 0]          # number of background/signal traces respectively
        self.__file_count = 0           # number of calls to __getitem__ per epoch
        self.__epochs = 0               # number of epochs that the generator was used for
        
        self.__working_directory = "/cr/users/filip/data/first_simulation/tensorflow/signal/"

        if self.train:
            self.__signal_files = os.listdir(self.__working_directory)[:self.split]
        elif not self.train:
            self.__signal_files = os.listdir(self.__working_directory)[-self.split:]

    # generator that creates one batch, loads one signal file (~2000 events)
    # and randomly shuffles background events inbetween the signal traces
    def __getitem__(self, index) -> tuple :

        self.__traces, self.__labels = [], []
        self.__file_count += 1

        # generate mixed set of signals and baselines
        events = np.loadtxt(self.__working_directory + self.__signal_files[index])

        for signal in events:

            # decide whether baseline gets interlaced
            choice = np.random.choice([0,1])

            while not choice:
                self.add_event(choice, signal)      # add background event to list
                choice = np.random.choice([0,1])    # add another background event?
            self.add_event(choice, signal)          # add signal event to list

        self.verbose and print("[" + self.__file_count * "-" + (self.__len__() - self.__file_count) * " " + f"] {self.__file_count}/{self.__len__()}" )

        # true batch size is not EXACTLY batch size, this should be okay
        return (np.array(self.__traces), np.array(self.__labels))

    # split into validation set and training set repectively
    def __len__(self):
        return self.split

    # add labelled trace consisting of either just baseline or baseline + signal             
    def add_event(self, choice : int, signal : np.ndarray) -> tuple:

        self.n_events[choice] += 1
        Baseline = VEMTrace(self.input_shape)
        if choice == 1: Baseline += signal

        choice_encoded = tf.keras.utils.to_categorical(choice, 2, dtype = int)
        self.__traces.append(Baseline.get())
        self.__labels.append(choice_encoded)

    # called by model.fit at the end of each epoch
    def on_epoch_end(self) -> None : 

        self.verbose and print(f"\nEPOCH {str(self.__epochs).zfill(3)} DONE; {self.n_events} events in buffer " + 100 * "_" + "\n")

        self.shuffle and np.random.shuffle(self.__signal_files)
        self.__file_count = 0
        self.n_events = [0, 0]
        self.__epochs += 1

    # fix random number generator seed for reproducibility
    @staticmethod
    def set_random_number_generator_seed(seed : int = 0) -> None :
        np.random.seed(seed)

# Wrapper for tf.keras.Sequential model with some additional functionalities
# TODO add __predict_batch, __predict_trace functionalities
class Classifier():

    def __init__(self, init_from_disk : str = None) -> None:

        if init_from_disk is None:

            self.__epochs = 0
            self.model = tf.keras.models.Sequential()

            # architecture of this NN is - apart from in/output - completely arbitrary, at least for now
            self.model.add(tf.keras.layers.Dense(units = 2048, input_shape=(trace_length,), activation = 'relu'))   # 1st hidden layer 2048 neurons
            self.model.add(tf.keras.layers.Dropout(0.2))                                                            # Dropout layer to fight overfitting
            self.model.add(tf.keras.layers.Dense(units = 12, activation = 'relu'))                                  # 2nd hidden layer 12 neurons
            self.model.add(tf.keras.layers.Dropout(0.2))                                                            # Dropout layer to fight overfitting
            self.model.add(tf.keras.layers.Dense(units = 2, activation = 'softmax', name = "Output"))               # Output layer with signal/background
        
        elif init_from_disk is not None:

            model_save_dir = "/cr/users/filip/data/first_simulation/tensorflow/model/"
            self.__epochs = int(init_from_disk[init_from_disk.rfind('_') + 1:])                                     # set previously run epochs as start
            self.model = tf.keras.models.load_model(model_save_dir + init_from_disk)                                # load model, doesn't work with h5py 3.x!

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model network on the provided training/validation set
    def train(self, training_set : TraceGenerator, validation_set : TraceGenerator, epochs : int) -> None:
        self.model.fit(training_set, validation_data=validation_set, initial_epoch = self.__epochs, epochs = epochs, verbose = 2)
        self.__epochs = epochs

    # Save the model to disk
    def save(self, directory_path : str) -> None : 
        self.model.save(directory_path + f"first_model_{self.__epochs}")

    # Predict a batch or single trace
    def predict(self):
        pass                # TODO

    # Wrapper for pretty printing
    def __str__(self) -> str :
        self.model.summary()
        return ""

if __name__ == "__main__":

    # Have 206563 station-level traces in dataset from first_simulation, see <check_signal_size.py> 
    n_signal = 206563               # total number of station-level traces in dataset
    trace_length = 20000            # total trace "duration", 8.3 ns/bin * 20 000 bins = 166 μs

    # initialize datasets (this doesn't load them into memory yet!)
    VirtualTrainingSet = TraceGenerator(train = True, split = 0.8, input_shape = trace_length, fix_seed = True, verbose = True)
    VirtualValidationSet = TraceGenerator(train = False, split = 0.2, input_shape = trace_length, fix_seed = True, verbose = False)

    # initialize convolutional neural network model
    SignalBackgroundClassifier = Classifier("first_model_250")

    # train the classifier and save it to disk
    SignalBackgroundClassifier.train(VirtualTrainingSet, VirtualValidationSet, 500)
    SignalBackgroundClassifier.save("/cr/users/filip/data/first_simulation/tensorflow/model/")