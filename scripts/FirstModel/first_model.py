#!/usr/bin/python3

from mimetypes import init
import typing
import sys, os
import numpy as np
import tensorflow as tf

# Baseline trace containing <trace_length> bins
# Default baseline: mean == 0 ADC; std == 1 ADC
class VEMTrace():

    def __init__(self, trace_length : int, mu : float = 0, std : float = 1) -> None :

        self.__array = np.random.normal(mu, std, trace_length)

    # overlay signal trace at uniformly random position of baseline
    def __iadd__(self, signal : np.array) -> object :
        assert trace_length > len(signal), "SIGNAL DOES NOT FIT INTO BASELINE!\n"

        start = np.random.randint(-trace_length, -len(signal))
        self.__array[start : start + len(signal)] += signal

        return self

    # default getter for baseline trace
    def get(self) -> np.array : 
        return self.__array

# Generator class for NN sequential model with some additional functionalities
# This website helped tremendously with writing a working example: shorturl.at/fFI09
class TraceGenerator(tf.keras.utils.Sequence):

    def __init__(self, train : bool, split : float, input_shape : int, fix_seed : bool = False, shuffle : bool = True) -> None:

        assert 0 < split and split < 1, "PLEASE PROVIDE A VALID SPLIT: 0 <= split <= 1"

        self.train = train              # wether object is used for training or validation
        self.split = split              # 0.00 - 1.00, percentage of data for this generator
        self.input_shape = input_shape  # input length of the data, trace duration = 8.3 ns * length
        self.fix_seed = fix_seed        # whether or not RNG seed is fixed, for reproducibility
        self.shuffle = shuffle          # whether or not to shuffle signal files at the end of each epoch

        self.n_events = [0, 0]          # number of background/signal traces respectively
        
        self.__working_directory = "/cr/users/filip/data/first_simulation/tensorflow/signal/"
        self.__signal_files = os.listdir(self.__working_directory)

    # generator that creates one batch, loads one signal file (~2000 events)
    # and randomly shuffles background events inbetween the signal traces
    def __getitem__(self, index) -> tuple :

        self.__traces, self.__labels = [], []

        # generate mixed set of signals and baselines
        events = np.loadtxt(self.__working_directory + self.__signal_files[index])

        for signal in events:

            # decide whether baseline gets interlaced
            choice = np.random.choice([0,1])

            while not choice:
                self.add_event(choice, signal)      # add background event to list
                choice = np.random.choice([0,1])    # add another background event?
            self.add_event(choice, signal)          # add signal event to list

        # true batch size is not EXACTLY batch size, this should be okay
        return (np.array(self.__traces), np.array(self.__labels))

    # have 100 signal files => number of steps per epoch = 100
    # such that optimizer sees all data over an entire epoch
    def __len__(self):
        return 100

    # add labelled trace consisting of either just baseline or baseline + signal             
    def add_event(self, choice : int, signal : np.ndarray) -> tuple:

        self.n_events[choice] += 1
        Baseline = VEMTrace(trace_length)
        if choice == 1: Baseline += signal

        choice_encoded = tf.keras.utils.to_categorical(choice, 2)
        self.__traces.append(Baseline.get())
        self.__labels.append(choice_encoded)

    # called by model.fit at the end of each epoch
    def on_epoch_end(self) -> None : 
        self.shuffle and np.random.shuffle(self.__signal_files)

    # fix random number generator seed for reproducibility
    @staticmethod
    def set_random_number_generator_seed(seed : int = 0) -> None :
        np.random.seed(seed)

# Wrapper for tf.keras.Sequential model with some additional functionalities
# TODO add __predict_batch, __predict_trace functionalities
class Classifier():

    def __init__(self, init_from_disk : str = None) -> None:

        self.epochs = 0
        self.model = tf.keras.models.Sequential()

        # architecture of this NN is - apart from in/output - completely arbitrary, at least for now
        # self.model.add(tf.keras.layers.Input(shape=(trace_length, ), dtype = tf.float32, name = "Input"))         # Input layer, trace_length neurons
        self.model.add(tf.keras.layers.Dense(units = 2048, input_shape=(trace_length,), activation = 'relu'))       # 1st hidden layer 2048 neurons
        self.model.add(tf.keras.layers.Dropout(0.2))                                                                # Dropout layer to fight overfitting
        self.model.add(tf.keras.layers.Dense(units = 12, activation = 'relu'))                                      # 2nd hidden layer 12 neurons
        self.model.add(tf.keras.layers.Dropout(0.2))                                                                # Dropout layer to fight overfitting
        self.model.add(tf.keras.layers.Dense(units = 2, activation = 'softmax', name = "Output"))                   # Output layer with signal/background
        
        if init_from_disk is not None:
            self.epochs = int(init_from_disk[init_from_disk.rfind('_') + 1:init_from_disk.rfind('.')])              # set previously run epochs as start
            self.model = tf.keras.models.load_model(init_from_disk)                                                 # load model, doesn't work with h5py 3.x !!
        
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model network on the provided training/validation set
    def train(self, training_set : TraceGenerator, validation_set : TraceGenerator, epochs : int, verbose : int = 0) -> None:
        
        # clear terminal for better overview of verbose output:
        verbose and print("\x1B[2J\x1B[H")

        self.epochs += epochs
        self.model.fit(training_set, validation_data=validation_set, epochs = epochs, verbose = verbose)

    # Save the model to disk
    def save(self, directory_path : str) -> None : 
        self.model.save(directory_path + f"first_model_{self.epochs}.SavedModel")

    # Predict a batch or single trace
    def predict(self):
        pass                # TODO

    # Wrapper for pretty printing
    def __str__(self) -> str :
        self.model.summary()
        return ""

if __name__=="__main__":

    # Have 206563 station-level traces in dataset from first_simulation, see <check_signal_size.py> 

    n_signal = 206563               # total number of station-level traces in dataset
    trace_length = 20000            # total trace "duration", 8.3 ns/bin * 20 000 bins = 166 μs

    # initialize datasets (this doesn't load them into memory yet!)
    VirtualTrainingSet = TraceGenerator(train = True, split = 0.8, input_shape = trace_length, fix_seed = True)
    VirtualValidationSet = TraceGenerator(train = True, split = 0.2, input_shape = trace_length, fix_seed = True)

    # initialize convolutional neural network model
    SignalBackgroundClassifier = Classifier()

    # train the classifier and save it to disk
    SignalBackgroundClassifier.train(VirtualTrainingSet, VirtualValidationSet, 100, 1)
    SignalBackgroundClassifier.save("/cr/users/filip/data/first_simulation/tensorflow/model/")