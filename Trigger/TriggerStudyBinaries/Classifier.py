import typing, sys, os
import tensorflow as tf
import numpy as np

# custom modules for specific use case
from TriggerStudyBinaries.Signal import VEMTrace
from TriggerStudyBinaries.Generator import EventGenerator

class NeuralNetworkArchitectures():

    ### Library functions to add layers #################
    if True: # so that this can be collapsed in editor =)
        @staticmethod
        def add_input(cls, **kwargs) -> typing.NoReturn :
            cls.model.add(tf.keras.layers.Input(**kwargs))

        @staticmethod
        def add_dense(cls, **kwargs) -> typing.NoReturn : 
            cls.model.add(tf.keras.layers.Dense(**kwargs))

        @staticmethod
        def add_conv1d(cls, **kwargs) -> typing.NoReturn : 
            cls.model.add(tf.keras.layers.Conv1D(**kwargs))

        @staticmethod
        def add_conv2d(cls, **kwargs) -> typing.NoReturn : 
            cls.model.add(tf.keras.layers.Conv2D(**kwargs))

        @staticmethod
        def add_flatten(cls, **kwargs) -> typing.NoReturn : 
            cls.model.add(tf.keras.layers.Flatten(**kwargs))

        @staticmethod
        def add_output(cls, **kwargs) -> typing.NoReturn : 
            cls.model.add(tf.keras.layers.Flatten())
            cls.model.add(tf.keras.layers.Dense(**kwargs))

        @staticmethod
        def add_dropout(cls, **kwargs) -> typing.NoReturn : 
            cls.model.add(tf.keras.layers.Dropout(**kwargs))
    #####################################################

    @staticmethod
    def __minimal_conv2d__(cls : "NNClassifier") -> typing.NoReturn :

        NeuralNetworkArchitectures.add_input(cls, shape = (3, 120, 1))
        NeuralNetworkArchitectures.add_conv2d(cls, filters = 2, kernel_size = 3, strides = 3)
        NeuralNetworkArchitectures.add_output(cls, units = 2, activation = "softmax")

# Wrapper for tf.keras.Sequential model with some additional functionalities
class NNClassifier(NeuralNetworkArchitectures):

    def __init__(self, set_architecture = None) -> typing.NoReturn :

        r'''
        :set_architecture ``None``: one of
        
        ``str`` -- path to existing network (relative to /)
        ``callable`` -- examples of architectures, see NeuralNetworkArchitectures
        '''

        super().__init__()

        if isinstance(set_architecture, typing.Callable):
            self.model = tf.keras.models.Sequential()
            set_architecture(self)
            self.epochs = 0
        elif isinstance(set_architecture, str):
            self.model = tf.keras.models.load_model("/cr/data01/filip/models/" + set_architecture)
            self.epochs = int(set_architecture[-1])

        self.model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = 'accuracy', run_eagerly = True)
        self.model.build()
        print(self)

    def train(self, Datasets : tuple, epochs : int, **kwargs) -> typing.NoReturn :
        
        TrainingSet, ValidationSet = Datasets

        try:
            verbosity = kwargs["verbose"]
        except KeyError:
            verbosity = 2
        
        self.model.fit(TrainingSet, validation_data = ValidationSet, epochs = epochs, verbose = verbosity)
        self.epochs += epochs
        self._signals, self._backgrounds = TrainingSet._signals, TrainingSet._backgrounds

    def save(self, directory_path : str) -> typing.NoReturn : 
        self.model.save("/cr/data01/filip/models/" + directory_path + f"model_{self.epochs}")

        with open("/cr/data01/filip/models/" + directory_path + f"model_{self.epochs}/statistics.txt", "w") as statistics:
            statistics.write(f"n_sig\tn_bkg\n{self._signals}\t{self._backgrounds}")

    def __call__(self, signal : np.ndarray) -> bool :

        # True if the network thinks it's seeing a signal
        # False if the network things it's not seeing a signal 

        # return np.argmax(self.model.__call__(np.reshape(signal(), (1, len(signal())))).numpy()[0]) == 1
        return self.model.__call__( tf.expand_dims(signal, axis = -1) )
        

    def __str__(self) -> str :
        self.model.summary()
        return ""

    def convert_to_C(self, save_file : str) -> typing.NoReturn :

        # TODO
        pass

    def add(self, layer : str, **kwargs) -> typing.NoReturn :
        print(self.layers[layer], layer, kwargs)
        self.layers[layer](**kwargs)

# Wrapper for currently employed station-level triggers (T1, T2, ToT, etc.)
# Information on magic numbers comes from Davids Mail on 03.03.22 @ 12:30pm
class TriggerClassifier():

    def __call__(self, trace : VEMTrace) -> bool : 
        return self.has_triggered(trace)

    # Whether or not any of the existing triggers caught this event
    def has_triggered(self, trace: VEMTrace) -> bool : 

        # Threshold of 3.2 immediately gets promoted to T2
        # Threshold of 1.75 if a T3 has already been issued
        T1_is_active = self.absolute_threshold_trigger(1.75, trace)

        return T1_is_active or self.time_over_threshold_trigger(trace)

    # method to check for (coincident) absolute signal threshold
    def absolute_threshold_trigger(self, threshold : float, signal : VEMTrace) -> bool : 

        # hierarchy doesn't (shouldn't?) matter
        for i in range(signal.trace_length):
            if signal.pmt_1[i] >= threshold:
                if signal.pmt_2[i] >= threshold:
                    if signal.pmt_3[i] >= threshold:
                        return True
                    else: continue
                else: continue
            else: continue
        
        return False

    # method to check for elevated baseline threshold trigger
    def time_over_threshold_trigger(self, signal : VEMTrace) -> bool : 

        window_length = 120      # amount of bins that are being checked
        threshold     = 0.2      # bins above this threshold are 'active'

        # count initial active bins
        pmt1_active = len(signal.pmt_1[:window_length][signal.pmt_1[:window_length] > threshold])
        pmt2_active = len(signal.pmt_2[:window_length][signal.pmt_2[:window_length] > threshold])
        pmt3_active = len(signal.pmt_3[:window_length][signal.pmt_3[:window_length] > threshold])

        for i in range(window_length, signal.trace_length):

            # check if ToT conditions are met
            ToT_trigger = [pmt1_active >= 13, pmt2_active >= 13, pmt3_active >= 13]

            if ToT_trigger.count(True) >= 2:
                return True

            # overwrite oldest bin and reevaluate
            pmt1_active += self.update_bin_count(i, signal.pmt_1, window_length, threshold)
            pmt2_active += self.update_bin_count(i, signal.pmt_2, window_length, threshold)
            pmt3_active += self.update_bin_count(i, signal.pmt_3, window_length, threshold)

        return False

    @staticmethod
    # helper method for time_over_threshold_trigger
    def update_bin_count(index : int, array: np.ndarray, window_length : int, threshold : float) -> int : 

        # is new bin active?
        if array[index] >= threshold:
            updated_bin_count = 1
        else:
            updated_bin_count = 0

        # was old bin active?
        if array[index - window_length] >= threshold:
            updated_bin_count -= 1

        return updated_bin_count

class BayesianClassifier():
    
    # TODO
    pass