from time import strftime, gmtime

from .__config__ import *
from .Signal import *
from .Generator import *


class NeuralNetworkArchitectures():

    ### Library functions to add layers #################
    if True: # so that this can be collapsed in editor =)
        @staticmethod
        def add_input(cls, **kwargs) -> None :
            cls.model.add(tf.keras.layers.Input(**kwargs))

        @staticmethod
        def add_dense(cls, **kwargs) -> None : 
            cls.model.add(tf.keras.layers.Dense(**kwargs))

        @staticmethod
        def add_conv1d(cls, **kwargs) -> None : 
            cls.model.add(tf.keras.layers.Conv1D(**kwargs))

        @staticmethod
        def add_conv2d(cls, **kwargs) -> None : 
            cls.model.add(tf.keras.layers.Conv2D(**kwargs))

        @staticmethod
        def add_flatten(cls, **kwargs) -> None : 
            cls.model.add(tf.keras.layers.Flatten(**kwargs))

        @staticmethod
        def add_output(cls, **kwargs) -> None : 
            cls.model.add(tf.keras.layers.Flatten())
            cls.model.add(tf.keras.layers.Dense(**kwargs))

        @staticmethod
        def add_dropout(cls, **kwargs) -> None : 
            cls.model.add(tf.keras.layers.Dropout(**kwargs))

        @staticmethod
        def add_norm(cls, **kwargs) -> None : 
            cls.model.add(tf.keras.layers.BatchNormalization(**kwargs))
    #####################################################

    @staticmethod   # 92 parameters
    def __one_layer_conv2d__(cls : "NNClassifier") -> None :

        NeuralNetworkArchitectures.add_input(cls, shape = (3, 120, 1))
        NeuralNetworkArchitectures.add_conv2d(cls, filters = 1, kernel_size = 3, strides = 3)
        NeuralNetworkArchitectures.add_output(cls, units = 2, activation = "softmax")


    @staticmethod   # 55 parameters
    def __two_layer_conv2d__(cls : "NNClassifier") -> None :

        NeuralNetworkArchitectures.add_input(cls, shape = (3, 120, 1))
        NeuralNetworkArchitectures.add_conv2d(cls, filters = 1, kernel_size = 3, strides = 3)
        NeuralNetworkArchitectures.add_conv1d(cls, filters = 1, kernel_size = 2, strides = 2)
        NeuralNetworkArchitectures.add_output(cls, units = 2, activation = "softmax")
    

    @staticmethod   # 35 parameters
    def __minimal_conv2d__(cls : "NNClassifier") -> None :

        NeuralNetworkArchitectures.add_input(cls, shape = (3, 120,1))
        NeuralNetworkArchitectures.add_conv2d(cls, filters = 2, kernel_size = (3,2), strides = 2)
        NeuralNetworkArchitectures.add_conv1d(cls, filters = 1, kernel_size = 2, strides = 2)
        NeuralNetworkArchitectures.add_conv1d(cls, filters = 1, kernel_size = 3, strides = 3)
        NeuralNetworkArchitectures.add_conv1d(cls, filters = 1, kernel_size = 3, strides = 3)
        NeuralNetworkArchitectures.add_output(cls, units = 2, activation = "softmax")

    @staticmethod   # 606 parameters
    def __large_conv2d__(cls : "NNClassifier") -> None : 

        NeuralNetworkArchitectures.add_input(cls, shape = (3, 120,1))
        NeuralNetworkArchitectures.add_conv2d(cls, filters = 2, kernel_size = (3,1), strides = 2)
        NeuralNetworkArchitectures.add_conv1d(cls, filters = 4, kernel_size = 3, strides = 3)
        NeuralNetworkArchitectures.add_conv1d(cls, filters = 8, kernel_size = 3, strides = 3)
        NeuralNetworkArchitectures.add_conv1d(cls, filters = 16, kernel_size = 3, strides = 3)
        NeuralNetworkArchitectures.add_output(cls, units = 2, activation = "softmax")

# Wrapper for tf.keras.Sequential model with some additional functionalities
class NNClassifier(NeuralNetworkArchitectures):


    models = \
        {
            "one_layer_conv2d" : NeuralNetworkArchitectures.__one_layer_conv2d__,
            "one_layer_conv2d_0.5VEM" : NeuralNetworkArchitectures.__one_layer_conv2d__,
            "one_layer_conv2d_1.0VEM" : NeuralNetworkArchitectures.__one_layer_conv2d__,
            "one_layer_conv2d_2.0VEM" : NeuralNetworkArchitectures.__one_layer_conv2d__,
            "two_layer_conv2d" : NeuralNetworkArchitectures.__two_layer_conv2d__,
            "minimal_conv2d" : NeuralNetworkArchitectures.__minimal_conv2d__,
            "large_conv2d" : NeuralNetworkArchitectures.__large_conv2d__
        }

    def __init__(self, set_architecture = None, supress_print : bool = False) -> None :

        r'''
        :set_architecture ``str``: one of
        
        -- path to existing network (relative to /cr/data01/filip/models/)
        -- examples of architectures, see model dictionary above 
        '''

        super().__init__()

        try:
            self.model = tf.keras.Sequential()
            self.models[set_architecture](self)
            self.epochs = 0
        except KeyError:
            try:
                self.model = tf.keras.models.load_model("/cr/data01/filip/models/" + set_architecture)
                self.epochs = int(set_architecture[-1])
            except OSError:
                sys.exit(f"\nCouldn't find path: '/cr/data01/filip/models/{set_architecture}', exiting now\n")

        self.model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = 'accuracy', run_eagerly = True)
        self.model.build()
        not supress_print and print(self)

    def train(self, Datasets : tuple, epochs : int, save_dir : str, **kwargs) -> None :
        
        TrainingSet, ValidationSet = Datasets
        verbosity = kwargs.get("verbose", 2)

        for i in range(self.epochs, epochs):
            print(f"Epoch {i + 1}/{epochs}")
            self.model.fit(TrainingSet, validation_data = ValidationSet, epochs = 1, verbose = verbosity)
            self.epochs += 1

            self.save(save_dir)

    def save(self, directory_path : str) -> None : 
        self.model.save("/cr/data01/filip/models/" + directory_path + f"/model_{self.epochs}")

    def __call__(self, signal : np.ndarray) -> bool :

        # 1 if the network thinks it's seeing a signal
        # 0 if the network things it's not seeing a signal 

        return np.array(self.model( tf.expand_dims([signal], axis = -1) )).argmax()        

    def __str__(self) -> str :
        self.model.summary()
        return ""

    def add(self, layer : str, **kwargs) -> None :
        print(self.layers[layer], layer, kwargs)
        self.layers[layer](**kwargs)

# Class for streamlined handling of multiple NNs with the same architecture
class Ensemble(NNClassifier):

    def __init__(self, set_architecture : str, n_models : int = 10) -> None :

        supress_print = False
        self.models = []

        try:

            # does this work?
            last_epoch = len(os.listdir("/cr/data01/filip/models/" + set_architecture + f"ensemble_1/")) - 1

            for i in range(1, n_models + 1):
                ThisModel = NNClassifier(set_architecture + f"ensemble_{i}/model_{last_epoch}", supress_print)
                self.models.append(ThisModel)

                supress_print = True

        except OSError:
            for i in range(n_models):

                ThisModel = NNClassifier(set_architecture, supress_print)
                self.models.append(ThisModel)

                supress_print = True

        self.name = set_architecture

        print(f"{self.name}: {n_models} models successfully initiated\n")

    def train(self, Datasets : tuple, epochs : int, **kwargs) -> None:

        start = perf_counter_ns()
        TrainingSet, ValidationSet = Datasets

        for i, instance in enumerate(self.models,1):

            elapsed = strftime('%H:%M:%S', gmtime((perf_counter_ns() - start)*1e-9))

            print(f"Model {i}/{len(self.models)}, {elapsed}s elapsed")

            for epoch in range(instance.epochs, epochs):
                instance.model.fit(TrainingSet, validation_data = ValidationSet, epochs = 1)
                instance.epochs = instance.epochs + 1

                instance.save(self.name + f"/ensemble_{i}/model_{epoch}")

        random.shuffle(TrainingSet.files)
        random.shuffle(ValidationSet.files)

    def __call__(self, trace : np.ndarray) -> list :

        return [model(trace) for model in self.models]


# Wrapper for currently employed station-level triggers (T1, T2, ToT, etc.)
# Information on magic numbers comes from Davids Mail on 03.03.22 @ 12:30pm
class TriggerClassifier():

    def __call__(self, trace : np.ndarray) -> int : 
        
        # Threshold of 3.2 immediately gets promoted to T2
        # Threshold of 1.75 if a T3 has already been issued

        if self.absolute_threshold_trigger(1.75, trace) or self.time_over_threshold_trigger(trace):
            return 1
        else: 
            return 0 

    # method to check for (coincident) absolute signal threshold
    def absolute_threshold_trigger(self, threshold : float, signal : np.ndarray) -> bool : 

        pmt_1, pmt_2, pmt_3 = signal

        # hierarchy doesn't (shouldn't?) matter
        for i in range(signal.shape[1]):
            if pmt_1[i] >= threshold:
                if pmt_2[i] >= threshold:
                    if pmt_3[i] >= threshold:
                        return True
                    else: continue
                else: continue
            else: continue
        
        return False

    # method to check for elevated baseline threshold trigger
    def time_over_threshold_trigger(self, signal : np.ndarray) -> bool : 

        threshold     = 0.2      # bins above this threshold are 'active'

        pmt_1, pmt_2, pmt_3 = signal

        # count initial active bins
        pmt1_active = list(pmt_1 > threshold).count(True)
        pmt2_active = list(pmt_2 > threshold).count(True)
        pmt3_active = list(pmt_3 > threshold).count(True)
        ToT_trigger = [pmt1_active >= 13, pmt2_active >= 13, pmt3_active >= 13]

        if ToT_trigger.count(True) >= 2:
            return True
        else:
            return False

class BayesianClassifier():
    
    # TODO
    pass