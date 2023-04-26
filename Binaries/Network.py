from .__config__ import *
from .Signal import *
from .Generator import *
from .Classifier import *

# Collection of different network architectures
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

        @staticmethod
        def add_flatten(model) -> None:
            model.add(tf.keras.layers.Flatten())

        @staticmethod
        def add_lstm(**kwargs) -> None :

            # doesn't work, since data is 2-dimensional
            # model.add(tf.keras.layers.LSTM(**kwargs))

            # instead use LSTM for each PMT
            input = tf.keras.layers.Input(kwargs.get("input_shape"))
            unstacked = tf.keras.layers.Lambda(lambda x: tf.unstack(x, axis=1))(input)
            dense_outputs = [tf.keras.layers.LSTM(kwargs.get("d_LSTM", 1), activation = "relu")(x) for x in unstacked]
            merged = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(dense_outputs)
            merged_flatten = tf.keras.layers.Flatten()(merged)

            return tf.keras.Model(input, tf.keras.layers.Dense(2, activation = "softmax")(merged_flatten))

    #####################################################

    # 44 parameters
    def __simple_LSTM__(self) -> tf.keras.Model : 
        return self.add_lstm(input_shape = (3, 120, 1), d_LSTM = 1)

    # doesn't really work all well with the dataset log E = 16-16.5 
    # since empty files raise background traces, which get scaled UP
    # 366 parameters
    def __normed_one_layer_conv2d__(self, model) -> None :

        self.add_input(model, shape = (3, 120, 1))
        self.add_norm(model)
        self.add_conv2d(model, filters = 4, kernel_size = 3, strides = 3)
        self.add_output(model, units = 2, activation = "softmax")

    # 92 parameters
    def __one_layer_conv2d__(self, model) -> None :

        self.add_input(model, shape = (3, 120, 1))
        self.add_conv2d(model, filters = 1, kernel_size = 3, strides = 3)
        self.add_output(model, units = 2, activation = "softmax")

    # 338 parameters
    def __one_large_layer_conv2d__(self, model) -> None : 

        self.add_input(model, shape = (3, 120, 1))
        self.add_conv2d(model, filters = 4, kernel_size = (3,1), strides = 3)
        self.add_output(model, units = 2, activation = "softmax")    

    # 1002 parameters
    def __one_layer_downsampling_equal__(self, model) -> None :

        self.add_input(model, shape = (3, 360, 1))
        self.add_conv2d(model, filters = 4, kernel_size = 3, strides = 3)
        self.add_output(model, units = 2, activation = "softmax")

    # 140 parameters
    def __two_layer_conv2d__(self, model) -> None :

        self.add_input(model, shape = (3, 120, 1))
        self.add_conv2d(model, filters = 4, kernel_size = 3, strides = 3)
        self.add_conv1d(model, filters = 2, kernel_size = 2, strides = 2)
        self.add_output(model, units = 2, activation = "softmax")

    # 630 parameters
    def __two_layer_downsampling_equal__(self, model) -> None :

        self.add_input(model, shape = (3, 360, 1))
        self.add_conv2d(model, filters = 8, kernel_size = 3, strides = 3)
        self.add_conv1d(model, filters = 4, kernel_size = 2, strides = 2)
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
        self.add_conv2d(model, filters = 4, kernel_size = (3,1), strides = 2)
        self.add_conv1d(model, filters = 8, kernel_size = 3, strides = 3)
        self.add_conv1d(model, filters = 16, kernel_size = 3, strides = 3)
        self.add_conv1d(model, filters = 4, kernel_size = 3, strides = 3)
        self.add_output(model, units = 2, activation = "softmax")

    def __N_CNN__(self, model, input_length) -> None : 

        self.add_input(model, shape = (3, input_length, 1))
        self.add_conv2d(model, filters = 4, kernel_size = 3, strides = 3)
        self.add_conv1d(model, filters = 2, kernel_size = 5, strides = 1)
        self.add_flatten(model)
        self.add_dense(model, units = 10, activation = "relu")
        self.add_dense(model, units = 2, activation = "softmax")

    # 294 parameters
    def __40_CNN__(self, model) -> None : 
        self.__N_CNN__(self, model, 40)

    # 434 parameters
    def __60_CNN__(self, model) -> None : 
        self.__N_CNN__(self, model, 60)

    # 634 parameters
    def __90_CNN__(self, model) -> None : 
        self.__N_CNN__(self, model, 90)

    # 834 parameters
    def __120_CNN__(self, model) -> None : 
        self.__N_CNN__(self, model, 120)

    # 1634 parameters
    def __240_CNN__(self, model) -> None : 
        self.__N_CNN__(self, model, 240)

    # 858 parameters
    def __kernel_10__(self, model) -> None :

        self.add_input(model, shape = (3, 120, 1))
        self.add_conv2d(model, filters = 4, kernel_size = (3, 10), strides = 3)
        self.add_conv1d(model, filters = 2, kernel_size = 5, strides = 1)
        self.add_flatten(model)
        self.add_dense(model, units = 10, activation = "relu")
        self.add_dense(model, units = 2, activation = "softmax")

    # 978 parameters
    def __kernel_30__(self, model) -> None :

        self.add_input(model, shape = (3, 120, 1))
        self.add_conv2d(model, filters = 4, kernel_size = (3, 30), strides = 3)
        self.add_conv1d(model, filters = 2, kernel_size = 5, strides = 1)
        self.add_flatten(model)
        self.add_dense(model, units = 10, activation = "relu")
        self.add_dense(model, units = 2, activation = "softmax")

# Early stopping callback that gets evaluated at the end of each batch
class BatchwiseEarlyStopping(tf.keras.callbacks.Callback):

    def __init__(self, patience : int, acc_threshold : float) -> None :
        self.acc_threshold = acc_threshold
        self.patience = patience

    def __reset__(self) -> None : 
        self.current_patience = 0
        self.best_loss = np.Inf

    def on_batch_end(self, batch, logs : dict = None) -> None :

        if logs.get("accuracy") >= self.acc_threshold:
            current_loss = logs.get("loss")
            if np.less(current_loss, self.best_loss):
                self.best_loss = current_loss
                self.current_patience = 0
            else:
                self.current_patience += 1

                if self.current_patience >= self.patience: raise EarlyStoppingError

    def on_train_begin(self, logs : dict = None) -> None : self.__reset__()
    def on_epoch_end(self, epoch, logs : dict = None) -> None : 
        self.__reset__()
        print()


# Wrapper for tf.keras.Sequential model with some additional functionalities
class NNClassifier(Classifier):    

    models = \
        {
            "one_layer_downsampling_equal" : Architectures.__one_layer_downsampling_equal__,
            "two_layer_downsampling_equal" : Architectures.__two_layer_downsampling_equal__,
            "normed_one_layer_conv2d" : Architectures.__normed_one_layer_conv2d__,
            "one_layer_conv2d" : Architectures.__one_layer_conv2d__,
            "one_large_layer_conv2d" : Architectures.__one_large_layer_conv2d__,
            "two_layer_conv2d" : Architectures.__two_layer_conv2d__,
            "minimal_conv2d" : Architectures.__minimal_conv2d__,
            "large_conv2d" : Architectures.__large_conv2d__,
            "simple_LSTM" : Architectures.__simple_LSTM__,
            "240_CNN" : Architectures.__240_CNN__,
            "120_CNN" : Architectures.__120_CNN__,
            "90_CNN" : Architectures.__90_CNN__,
            "60_CNN" : Architectures.__60_CNN__,
            "40_CNN" : Architectures.__40_CNN__,
            "60_CNN" : Architectures.__60_CNN__,
            "kernel_10" : Architectures.__kernel_10__,
            "kernel_30" : Architectures.__kernel_30__,
        }

    def __init__(self, name : str, set_architecture = None, supress_print : bool = False) -> None :

        r'''
        :name ``str``: specifies the name of the NN and directory in /cr/data01/filip/models/
        :set_architecture ``str``: specifies the architecture (upon first initialization)

        :Keyword arguments:
        
        * *early_stopping_patience* (``int``) -- number of batches without improvement before training is stopped
        '''

        super().__init__(name)

        if set_architecture is None:

            try:
                self.model = tf.keras.models.load_model("/cr/data01/filip/models/" + name + "/model_converged")
                self.epochs = "converged"
            except OSError:
                available_models = os.listdir('/cr/data01/filip/models/' + name)
                if len(available_models) == 0: raise FileNotFoundError
                elif len(available_models) != 1:
                    choice = input(f"\nSelect epoch from {available_models}\n Model: ")
                else: choice = available_models[0]
                self.model = tf.keras.models.load_model("/cr/data01/filip/models/" + name + "/" + choice)
                self.epochs = int(choice.split("_")[-1])
        else:
            try:
                self.model = tf.keras.Sequential()
                self.models[set_architecture](Architectures, self.model)
                self.model.build()
                
            # ValueError is raised for LSTM due to different initialization
            except TypeError:
                self.model = self.models[set_architecture](Architectures)

            self.epochs = 0

        self.model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'], run_eagerly = True)

        not supress_print and print(self)

    def train(self, Datasets : tuple, epochs : int, **kwargs) -> None :
        
        # stop if no improvement over 50% of epoch and accuracy over 95%
        early_stopping_patience = kwargs.get("early_stopping_patience", int(0.5 * len(Datasets[0])))
        early_stopping_accuracy = kwargs.get("early_stopping_accuracy", GLOBAL.early_stopping_accuracy)
        
        EarlyStopping = BatchwiseEarlyStopping(early_stopping_patience, early_stopping_accuracy)

        callbacks = [EarlyStopping,]

        try:
            os.makedirs(f"/cr/data01/filip/models/{self.name}")
        except FileExistsError: pass

        training_status = "normally"
        TrainingSet, ValidationSet = Datasets

        # print("Creating physics information for both datasets...")
        # TrainingSet.physics_test(n_showers = int(0.05 * TrainingSet.__len__()), save_dir = f"/cr/data01/filip/models/{self.name}/training_set_physics_test.png")
        # ValidationSet.physics_test(n_showers = int(0.2 * ValidationSet.__len__()), save_dir = f"/cr/data01/filip/models/{self.name}/validation_set_physics_test.png")

        try:
            self.model.fit(TrainingSet, validation_data = ValidationSet, epochs = epochs - self.epochs, callbacks = callbacks, verbose = kwargs.get("verbose", 2))
            self.epochs = epochs

        except EarlyStoppingError: 
            self.epochs = "converged"
            training_status = "early"

        self.save()

        with open(f"/cr/data01/filip/models/{self.name}/model_{self.epochs}/training_files.csv", "w") as training_file:
            for file in TrainingSet.files:
                training_file.write(file + "\n")

        with open(f"/cr/data01/filip/models/{self.name}/model_{self.epochs}/validation_files.csv", "w") as validation_file:
            for file in ValidationSet.files:
                validation_file.write(file + "\n")

        # provide some metadata
        print(f"\nTraining exited {training_status}. Onto providing metadata now...")
        self.make_signal_dataset(ValidationSet, f"validation_data")

        # calculate trigger rate on random traces
        window_step = TrainingSet.trace_options["window_step"]
        window_length = TrainingSet.trace_options["window_length"]

        print(f"\ncalculating trigger rate on 0.5s (~30 000 Traces) of random traces now")
        f, df, n, n_trig, t = self.production_test(30000, apply_downsampling = kwargs.get("apply_downsampling", False), window_length = window_length, window_step = window_step)

        with open(f"/cr/data01/filip/models/{self.name}/model_{self.epochs}/production_test.csv", "w") as random_file:
            random_file.write("# f  df  n_traces  n_total_triggered  total_trace_duration\n")
            random_file.write(f"{f} {df} {n} {n_trig} {t}")


    def save(self) -> None : 
        self.model.save(f"/cr/data01/filip/models/{self.name}/model_{self.epochs}")

    def __call__(self, signal : np.ndarray) -> typing.Union[bool, tuple] :

        # 1 if the network thinks it's seeing a signal
        # 0 if the network thinks it's seening background 

        if len(signal.shape) == 3:                                                  # predict on batch
            predictions = self.model.predict_on_batch(signal)

            return np.array([prediction.argmax() for prediction in predictions])

        elif len(signal.shape) == 2:                                                # predict on sample
            
            return np.array(self.model( tf.expand_dims([signal], axis = -1) )).argmax()        

    def __str__(self) -> str :
        self.model.summary()
        return ""

    # add a layer to the model architecture
    def add(self, layer : str, **kwargs) -> None :
        print(self.layers[layer], layer, kwargs)
        self.layers[layer](**kwargs)

    # get validation/training file path in a list
    def get_files(self, dataset : str) -> typing.Union[list, None] : 

        if dataset not in ["training", "validation"]:
            print(f"[WARN] -- Attempt to fetch invalid dataset: <{dataset}>")
            return None
        
        return list(np.loadtxt(f"/cr/data01/filip/models/{self.name}/model_{self.epochs}/{dataset}_files.csv", dtype = str))