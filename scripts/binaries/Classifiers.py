import typing, sys, os
import tensorflow as tf
import numpy as np

# custom modules for specific use case
from binaries.EventGenerators import EventGenerator
from binaries.Signal import VEMTrace

# Wrapper for tf.keras.Sequential model with some additional functionalities
class NNClassifier():

    def __init__(self, set_architecture = None) -> typing.NoReturn :

        self.layers = {"Dense" : self.add_dense, "Conv1D" : self.add_conv1d, "Flatten" : self.add_flatten, 
        "Input" : self.add_input, "Dropout" : self.add_dropout, "Output" : self.add_output}

        tf.config.run_functions_eagerly(True)

        if isinstance(set_architecture, typing.Callable):
            self.model = tf.keras.models.Sequential()
            set_architecture(self)
            self.epochs = 0
        elif isinstance(set_architecture, str):
            self.model = tf.keras.models.load_model("/cr/data01/filip/" + set_architecture)
            self.epochs = int(set_architecture[-1])

        self.model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = 'accuracy', run_eagerly = True)
        print(self)

    def train(self, dataset : typing.Any, epochs : int, **kwargs) -> typing.NoReturn :
        
        TrainingSet, ValidationSet = EventGenerator(dataset, *kwargs)
        self.model.fit(TrainingSet, validation_data = ValidationSet, epochs = epochs, verbose = 2)
        self.epochs += epochs

    def save(self, directory_path : str) -> typing.NoReturn : 
        self.model.save("/cr/data01/filip/" + directory_path + f"model_{self.epochs}")

    def predict(self, trace : list) -> bool :

        # True if the network thinks it's seeing a signal
        # False if the network things it's not seeing a signal 

        return np.argmax(self.model.__call__(np.reshape(trace, (1, len(trace)))).numpy()[0]) == 1

    def __str__(self) -> str :
        self.model.summary()
        return ""
    
    def add_input(self, **kwargs) -> typing.NoReturn :
        self.model.add(tf.keras.layers.Input(**kwargs))

    def add_dense(self, **kwargs) -> typing.NoReturn : 
        self.model.add(tf.keras.layers.Dense(**kwargs))

    def add_conv1d(self, **kwargs) -> typing.NoReturn : 
        self.model.add(tf.keras.layers.Conv1D(**kwargs))

    def add_flatten(self, **kwargs) -> typing.NoReturn : 
        self.model.add(tf.keras.layers.Flatten(**kwargs))

    def add_output(self, **kwargs) -> typing.NoReturn : 
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(**kwargs))

    def add_dropout(self, **kwargs) -> typing.NoReturn : 
        self.model.add(tf.keras.layers.Dropout(**kwargs))


# TODO WRITE THIS
# Wrapper for currently employed station-level triggers (T1, T2, ToT, etc.)
class Trigger():

    

    # Whether or not any of the existing triggers caught this event
    def has_triggered(self, signal) -> bool : 

        # check T1 first, then ToT, for performance reasons
        # have T2 only when T1 also triggered, so ignore it
        T1_is_active = self.absolute_threshold_trigger(1.75, signal)

        if not T1_is_active:
            ToT_is_active = self.time_over_threshold_trigger()

        return T1_is_active or ToT_is_active

    # method to check for (coincident) absolute signal threshold
    def absolute_threshold_trigger(self, threshold : float, ) -> bool : 

        # hierarchy doesn't (shouldn't?) matter, since we need coincident signal anyway
        for i in range(self.trace_length):
            if self.__pmt_1[i] >= threshold:
                if self.__pmt_2[i] >= threshold:
                    if self.__pmt_3[i] >= threshold:
                        return True
                    else: continue
                else: continue
            else: continue
        
        return False

    # # method to check for elevated baseline threshold trigger
    # def time_over_threshold_trigger(self) -> bool : 

    #     window_length = 120      # amount of bins that are being checked
    #     threshold     = 0.2      # bins above this threshold are 'active'
        
    #     # count initial active bins
    #     pmt1_active = len(self.__pmt_1[:window_length][self.__pmt_1[:window_length] > threshold])
    #     pmt2_active = len(self.__pmt_2[:window_length][self.__pmt_2[:window_length] > threshold])
    #     pmt3_active = len(self.__pmt_3[:window_length][self.__pmt_3[:window_length] > threshold])

    #     for i in range(window_length, self.trace_length):

    #         # check if ToT conditions are met
    #         ToT_trigger = [pmt1_active >= 13, pmt2_active >= 13, pmt3_active >= 13]

    #         if ToT_trigger.count(True) >= 2:
    #             return True

    #         # overwrite oldest bin and reevaluate
    #         pmt1_active += self.update_bin_count(i, self.__pmt_1, window_length, threshold)
    #         pmt2_active += self.update_bin_count(i, self.__pmt_2, window_length, threshold)
    #         pmt3_active += self.update_bin_count(i, self.__pmt_3, window_length, threshold)

    #     return False

    # @staticmethod
    # # helper method for time_over_threshold_trigger
    # def update_bin_count(index : int, array: np.ndarray, window_length : int, threshold : float) -> int : 

    #     # is new bin active?
    #     if array[index] >= threshold:
    #         updated_bin_count = 1
    #     else:
    #         updated_bin_count = 0

    #     # was old bin active?
    #     if array[index - window_length] >= threshold:
    #         updated_bin_count -= 1

    #     return updated_bin_count