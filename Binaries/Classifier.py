from time import strftime, gmtime
from abc import abstractmethod

from .__config__ import *
from .Signal import *
from .Generator import *

class Classifier():

    @abstractmethod
    def __init__(self, name : str) -> None : self.name = name

    @abstractmethod
    def __call__(self) -> int : raise NotImplementedError

    def production_test(self, n_traces : int = GLOBAL.n_production_traces, **kwargs) -> None :

        start = perf_counter_ns()
        n_total_triggered = 0
        trigger_examples = []

        RandomTraces = EventGenerator(["19_19.5"], split = 1, force_inject = 0, real_background = True, prior = 0, **kwargs)
        RandomTraces.files = np.zeros(n_traces)

        for batch in range(RandomTraces.__len__()):

            elapsed = perf_counter_ns() - start
            mean_per_step_ms = elapsed / (batch + 1) * 1e-6

            print(f"{100 * (batch/n_traces):.2f}% - {mean_per_step_ms:.2f}ms/batch, ETA = {(n_traces - batch) * mean_per_step_ms * 1e-3:.0f}s          ", end ="\r")
            
            traces, _ = RandomTraces.__getitem__(batch, full_trace = True)
            trace = traces[0]
            
            for index in RandomTraces.__sliding_window__(trace):

                window, _ = trace.get_trace_window((index, index + RandomTraces.window_length), skip_integral = True)
                
                if self.__call__(window):

                    n_total_triggered += 1

                    if n_total_triggered < 10:
                        trigger_examples.append(trace)

                    # perhaps skipping the entire trace isn't exactly accurate
                    # but then again just skipping one window seems wrong also
                    break

        total_trace_duration = GLOBAL.single_bin_duration * GLOBAL.n_bins * n_traces
        trigger_frequency = n_total_triggered / total_trace_duration

        print("\n\nProduction test results:")
        print("")
        print(f"random traces injected: {n_traces}")
        print(f"total T2 trigger found: {n_total_triggered}")
        print(f"*********************************")
        print(f"TRIGGER FREQUENCY = {trigger_frequency:.4f} Hz")

        for trace in trigger_examples: trace.__plot__()

from .Testing import *

# Wrapper for tf.keras.Sequential model with some additional functionalities
class NNClassifier(Classifier):

    # Early stopping callback that gets evaluated at the end of each batch
    class BatchwiseEarlyStopping(tf.keras.callbacks.Callback):

        def __init__(self, patience : int) -> None :
            self.patience = patience
        
        def on_train_begin(self, logs : dict = None) -> None :
            
            self.current_patience = 0
            self.best_loss = np.Inf

        def on_batch_end(self, batch, logs : dict = None) -> None :

            if logs.get("accuracy") >= 0.99:
                current_loss = logs.get("loss")
                if np.less(current_loss, self.best_loss):
                    self.best_loss = current_loss
                    self.current_patience = 0
                else:
                    self.current_patience += 1

                    if self.current_patience >= self.patience: raise EarlyStoppingError

    class Architectures():

        ### Library functions to add layers #################
        if True: # so that this can be collapsed in editor =)
            @staticmethod
            def add_input(model, **kwargs) -> None :
                model.add(tf.keras.layers.Input(**kwargs))

            @staticmethod
            def add_dense(model, **kwargs) -> None : 
                model.add(tf.keras.layers.Dense(**kwargs))

            @staticmethod
            def add_conv1d(model, **kwargs) -> None : 
                model.add(tf.keras.layers.Conv1D(**kwargs))

            @staticmethod
            def add_conv2d(model, **kwargs) -> None : 
                model.add(tf.keras.layers.Conv2D(**kwargs))

            @staticmethod
            def add_flatten(model, **kwargs) -> None : 
                model.add(tf.keras.layers.Flatten(**kwargs))

            @staticmethod
            def add_output(model, **kwargs) -> None : 
                model.add(tf.keras.layers.Flatten())
                model.add(tf.keras.layers.Dense(**kwargs))

            @staticmethod
            def add_dropout(model, **kwargs) -> None : 
                model.add(tf.keras.layers.Dropout(**kwargs))

            @staticmethod
            def add_norm(model, **kwargs) -> None : 
                model.add(tf.keras.layers.BatchNormalization(**kwargs))
        #####################################################

        # doesn't really work all well with the dataset log E = 16-16.5 
        # since empty files raise background traces, which get scaled UP
        # 96 parameters
        def __normed_one_layer_conv2d__(self, model) -> None :

            self.add_input(model, shape = (3, 120, 1))
            self.add_norm(model)
            self.add_conv2d(model, filters = 1, kernel_size = 3, strides = 3)
            self.add_output(model, units = 2, activation = "softmax")

        # 92 parameters
        def __one_layer_conv2d__(self, model) -> None :

            self.add_input(model, shape = (3, 120, 1))
            self.add_conv2d(model, filters = 1, kernel_size = 3, strides = 3)
            self.add_output(model, units = 2, activation = "softmax")


        # 55 parameters
        def __two_layer_conv2d__(self, model) -> None :

            self.add_input(model, shape = (3, 120, 1))
            self.add_conv2d(model, filters = 1, kernel_size = 3, strides = 3)
            self.add_conv1d(model, filters = 1, kernel_size = 2, strides = 2)
            self.add_output(model, units = 2, activation = "softmax")
        

        # 35 parameters
        def __minimal_conv2d__(self, model) -> None :

            self.add_input(model, shape = (3, 120,1))
            self.add_conv2d(model, filters = 2, kernel_size = (3,2), strides = 2)
            self.add_conv1d(model, filters = 1, kernel_size = 2, strides = 2)
            self.add_conv1d(model, filters = 1, kernel_size = 3, strides = 3)
            self.add_conv1d(model, filters = 1, kernel_size = 3, strides = 3)
            self.add_output(model, units = 2, activation = "softmax")

        # 606 parameters
        def __large_conv2d__(self, model) -> None : 

            self.add_input(model, shape = (3, 120,1))
            self.add_conv2d(model, filters = 2, kernel_size = (3,1), strides = 2)
            self.add_conv1d(model, filters = 4, kernel_size = 3, strides = 3)
            self.add_conv1d(model, filters = 8, kernel_size = 3, strides = 3)
            self.add_conv1d(model, filters = 16, kernel_size = 3, strides = 3)
            self.add_output(model, units = 2, activation = "softmax")

    models = \
        {
            "normed_one_layer_conv2d" : Architectures.__normed_one_layer_conv2d__,
            "minimal_conv2d_0.00VEM_downsampled" : Architectures.__one_layer_conv2d__,
            "one_layer_conv2d_0.20VEM" : Architectures.__one_layer_conv2d__,
            "minimal_conv2d_0.20VEM_downsampled" : Architectures.__one_layer_conv2d__,
            "one_layer_conv2d_0.50VEM" : Architectures.__one_layer_conv2d__,
            "minimal_conv2d_0.50VEM_downsampled" : Architectures.__one_layer_conv2d__,
            "one_layer_conv2d_1.00VEM" : Architectures.__one_layer_conv2d__,
            "minimal_conv2d_1.00VEM_downsampled" : Architectures.__one_layer_conv2d__,
            "one_layer_conv2d_3.00VEM" : Architectures.__one_layer_conv2d__,
            "minimal_conv2d_3.00VEM_downsampled" : Architectures.__one_layer_conv2d__,
            "one_layer_conv2d" : Architectures.__one_layer_conv2d__,
            "two_layer_conv2d" : Architectures.__two_layer_conv2d__,
            "minimal_conv2d" : Architectures.__minimal_conv2d__,
            "large_conv2d" : Architectures.__large_conv2d__
        }

    def __init__(self, set_architecture = None, supress_print : bool = False, **kwargs) -> None :

        r'''
        :set_architecture ``str``: one of
        
        -- path to existing network (relative to /cr/data01/filip/models/)
        -- examples of architectures, see NeuralNetworkArchitectures

        :Keyword arguments:
        
        * *early_stopping_patience* (``int``) -- number of batches without improvement before training is stopped
        '''

        super().__init__(set_architecture)

        try:
            self.model = tf.keras.Sequential()
            self.models[set_architecture](self.Architectures, self.model)
            self.epochs = 0
        except KeyError:
            try:
                self.model = tf.keras.models.load_model("/cr/data01/filip/models/" + set_architecture)
                try: self.epochs = int(set_architecture[-1])
                except ValueError: self.epochs = 100
            except OSError:
                sys.exit(f"\nCouldn't find path: '/cr/data01/filip/models/{set_architecture}', exiting now\n")

        self.model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = [tf.keras.metrics.Precision(), 'accuracy'], run_eagerly = True)
        self.model.build()
        
        EarlyStopping = self.BatchwiseEarlyStopping(kwargs.get("early_stopping_patience", GLOBAL.early_stopping_patience))

        self.callbacks = [EarlyStopping,]
        
        not supress_print and print(self)

    def train(self, Datasets : tuple, epochs : int) -> None :
        
        TrainingSet, ValidationSet = Datasets

        try:
            for i in range(self.epochs, epochs):
                print(f"Epoch {i + 1}/{epochs}")
                self.history = self.model.fit(TrainingSet, validation_data = ValidationSet, epochs = 1, callbacks = self.callbacks)
                self.epochs += 1
        except EarlyStoppingError: 
            self.epochs = "converged"

        self.save(self.name)

        # provide some metadata
        print("\nTraining exited normally. Onto providing metadata now...")

        os.system(f"mkdir -p /cr/data01/filip/models/{self.name}/model_{self.epochs}/ROC_curve")
        true_positive_rate = make_dataset(self, ValidationSet, f"validation_data")

        with open(f"/cr/data01/filip/models/{self.name}/metrics.csv", "w") as metadata:
            for key, value in self.history.history.items():
                metadata.write(f"{key} {value[0]}\n")

            metadata.write(f"val_tpr {true_positive_rate}")


    def save(self, directory_path : str) -> None : 
        self.model.save(f"/cr/data01/filip/models/{directory_path}/model_{self.epochs}")

    def __call__(self, signal : np.ndarray) -> bool :

        # 1 if the network thinks it's seeing a signal
        # 0 if the network thinks it's seening background 

        return np.array(self.model( tf.expand_dims([signal], axis = -1) )).argmax()        

    def __str__(self) -> str :
        self.model.summary()
        return ""

    def add(self, layer : str, **kwargs) -> None :
        print(self.layers[layer], layer, kwargs)
        self.layers[layer](**kwargs)


# Class for streamlined handling of multiple NNs with the same architecture
class Ensemble(NNClassifier):

    def __init__(self, set_architecture : str, n_models : int = GLOBAL.n_ensembles) -> None :

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
                instance.model.fit(TrainingSet, validation_data = ValidationSet, epochs = 1, verbose = 0)
                instance.epochs = instance.epochs + 1

                instance.save(self.name + f"/ensemble_{i}/")

        random.shuffle(TrainingSet.files)
        random.shuffle(ValidationSet.files)

    def __call__(self, trace : np.ndarray) -> list :

        return [model(trace) for model in self.models]


# Wrapper for currently employed station-level triggers (T1, T2, ToT, etc.)
# Information on magic numbers comes from Davids Mail on 03.03.22 @ 12:30pm
class HardwareClassifier(Classifier):

    def __init__(self) : 
        super().__init__("HardwareClassifier")

    def __call__(self, trace : np.ndarray) -> bool : 
        
        # Threshold of 3.2 immediately gets promoted to T2
        # Threshold of 1.75 if a T3 has already been issued

        return self.Th(3.2, trace) or self.ToT(trace) or self.ToTd(trace) or self.MoPS(trace)

    # method to check for (coincident) absolute signal threshold
    def Th(self, threshold : float, signal : np.ndarray) -> bool : 

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
    def ToT(self, signal : np.ndarray) -> bool : 

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

    # method to check for elevated baseline of deconvoluted signal
    # first bin of trace is ignored, this shouldn't matter too much hopefully
    def ToTd(self, signal : np.ndarray) -> bool : 

        # for information on this see GAP note 2018-01
        dt      = 8.3                                                               # UUB bin width
        tau     = 67                                                                # decay constant
        decay   = np.exp(-dt/tau)                                                   # decay term
        deconvoluted_trace = []

        for pmt in signal:
            deconvoluted_pmt = [(pmt[i] - pmt[i-1] * decay)/(1 - decay) for i in range(1,len(pmt))]
            deconvoluted_trace.append(deconvoluted_pmt)
 
        return self.ToT(np.array(deconvoluted_trace))

    # method to count positive flanks in an FADC trace
    def MoPS(self, signal : np.ndarray) -> bool : 

        # as per GAP note 2018-01; an exact offline reconstruction of the trigger is not possible
        # Can this be fixed in some way? perhaps with modified integration threshold INT?
        return False 


class BayesianClassifier(Classifier):
    
    def __init__(self) -> None :

        super().__init__("BayesianClassifier")

    #     self.bin_centers = np.loadtxt("/cr/data01/filip/models/naive_bayes_classifier/bins.csv")
    #     self.signal = np.loadtxt("/cr/data01/filip/models/naive_bayes_classifier/signal.csv")
    #     self.background = np.loadtxt("/cr/data01/filip/models/naive_bayes_classifier/background.csv")
    #     self.quotient = self.signal / self.background
    #     self.threshold = threshold

    def __call__(self, trace : np.ndarray) -> bool : 
        
        # mock_likelihood = 0

        # for PMT in trace:

        #     bins = [self.find_bin(value) for value in PMT]
        #     mock_likelihood += np.sum(np.log(self.quotient[bins]))

        # return mock_likelihood > self.threshold

        return Trace.integrate(trace) > 1.5874890619840887

    # return the index of the bin (from self.bin_centers) that value would fall into
    def find_bin(self, value : int) -> int : 
        return np.abs(self.bin_centers - value).argmin()