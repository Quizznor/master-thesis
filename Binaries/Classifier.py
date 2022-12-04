from time import strftime, gmtime
from abc import abstractmethod
from datetime import datetime

from .__config__ import *
from .Signal import *
from .Generator import *

# Some general definitiones that all classifiers should have
class Classifier():

    @abstractmethod
    def __init__(self, name : str) -> None : self.name = name

    @abstractmethod
    def __call__(self) -> int : raise NotImplementedError

    # test the trigger rate of the classifier on random traces
    def production_test(self, n_traces : int = GLOBAL.n_production_traces, **kwargs) -> None :

        total_trace_duration = GLOBAL.n_bins * GLOBAL.single_bin_duration * n_traces
        self.start_time = str(datetime.now()).replace(" ", "-")[:-10]
        start = perf_counter_ns()
        n_total_triggered = 0

        os.system(f"mkdir -p /cr/users/filip/plots/production_tests/{self.name.replace('/','-')}/{self.start_time}/")

        RandomTraces = EventGenerator(["19_19.5"], split = 1, force_inject = 0, real_background = True, prior = 0, **kwargs)
        RandomTraces.files = np.zeros(n_traces)

        n_bins = GLOBAL.n_bins if not RandomTraces.downsampling else GLOBAL.n_bins // 3
        t_single_bin = GLOBAL.single_bin_duration if not RandomTraces.downsampling else GLOBAL.single_bin_duration * 3

        if type(self) == type(HardwareClassifier()):

            for batch in range(RandomTraces.__len__()):

                elapsed = perf_counter_ns() - start
                mean_per_step_ms = elapsed / (batch + 1) * 1e-6
                to_str = lambda x : f"{int(x//3600)}h {int((x % 3600)//60)}m"

                rate = n_total_triggered / ((batch + 1)/n_traces * total_trace_duration)
                print(f"{100 * (batch/n_traces):.2f}% - {mean_per_step_ms:.2f}ms/batch, ETA = {to_str((n_traces - batch) * mean_per_step_ms * 1e-3)} -> {n_total_triggered} trace(s) triggered, {rate:.2f} Hz      ", end ="\r")

                traces, _ = RandomTraces.__getitem__(batch, full_trace = True)
                trace = traces[0]

                for index in RandomTraces.__sliding_window__(trace):

                    window, _, _, _ = trace.get_trace_window((index, index + RandomTraces.window_length), skip_integral = True)
                    
                    if self.__call__(window):

                        n_total_triggered += 1

                        if n_total_triggered < 100: self.plot_trace_window(window, n_total_triggered)

                        # perhaps skipping the entire trace isn't exactly accurate
                        # but then again just skipping one window seems wrong also
                        # -> David thinks this should be fine
                        break

        else:

            for batch, (traces, _, _) in enumerate(RandomTraces):

                elapsed = perf_counter_ns() - start
                mean_per_step_ms = elapsed / (batch + 1) * 1e-6
                to_str = lambda x : f"{int(x//3600)}h {int((x % 3600)//60)}m"
                predictions = self.__call__(traces)

                print(f"{100 * (batch/n_traces):.2f}% - {mean_per_step_ms:.2f}ms/batch, ETA = {to_str((n_traces - batch) * mean_per_step_ms * 1e-3)} -> {n_total_triggered} trace(s) triggered          ", end ="\r")

                if np.any(predictions):
                    n_total_triggered += 1

                    if n_total_triggered < 100: self.plot_trace_window(traces[np.argmax(predictions)], n_total_triggered)               # plots the FIRST trigger in batch!    

        total_trace_duration = t_single_bin * n_bins * n_traces
        trigger_frequency = n_total_triggered / total_trace_duration

        print("\n\nProduction test results:")
        print("")
        print(f"random traces injected: {n_traces}")
        print(f"summed traces duration: {total_trace_duration:.4}s")
        print(f"total T2 trigger found: {n_total_triggered}")
        print(f"*********************************")
        print(f"TRIGGER FREQUENCY = {trigger_frequency:.4f} Hz")

    def plot_trace_window(self, trace : np.ndarray, index : int) -> None : 

        (pmt1, pmt2, pmt3), x = trace, len(trace[0])
        assert len(pmt1) == len(pmt2) == len(pmt3), "TRACE LENGTHS DO NOT MATCH"

        plt.plot(range(x), pmt1, label = "PMT #1")
        plt.plot(range(x), pmt2, label = "PMT #2")
        plt.plot(range(x), pmt3, label = "PMT #3")

        plt.ylabel(r"Signal / VEM$_\mathrm{Peak}$")
        plt.xlabel("Bin / 8.33 ns")
        plt.xlim(0, x)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(f"/cr/users/filip/plots/production_tests/{self.name.replace('/','-')}/{self.start_time}/trigger_{index}")
        plt.cla()


    # Performance visualizers #######################################################################
    if True:                                              # So this can be collapsed in the editor =)
        # load a specific dataset (e.g. 'validation_data', 'real_background', etc.) and print performance
        def load_and_print_performance(self, dataset : str, usecols : list = None) -> tuple : 

            try:
                if header_was_called: pass

            except NameError:
                self.__header__()

            print(f"Fetching predictions for: {self.name} -> {dataset}", end = "\r")

            try:
                save_files = \
                {
                    "TP" : f"/cr/data01/filip/models/{self.name}/ROC_curve/{dataset}/true_positives.csv",
                    "TN" : f"/cr/data01/filip/models/{self.name}/ROC_curve/{dataset}/true_negatives.csv",
                    "FP" : f"/cr/data01/filip/models/{self.name}/ROC_curve/{dataset}/false_positives.csv",
                    "FN" : f"/cr/data01/filip/models/{self.name}/ROC_curve/{dataset}/false_negatives.csv"
                }

                TP = np.loadtxt(save_files['TP'])
                FP = np.loadtxt(save_files['FP'])
                TN = np.loadtxt(save_files['TN'])
                FN = np.loadtxt(save_files['FN'])

            except OSError:

                save_files = \
                {
                    "TP" : f"/cr/data01/filip/models/{self.name}/model_{self.epochs}/ROC_curve/{dataset}/true_positives.csv",
                    "TN" : f"/cr/data01/filip/models/{self.name}/model_{self.epochs}/ROC_curve/{dataset}/true_negatives.csv",
                    "FP" : f"/cr/data01/filip/models/{self.name}/model_{self.epochs}/ROC_curve/{dataset}/false_positives.csv",
                    "FN" : f"/cr/data01/filip/models/{self.name}/model_{self.epochs}/ROC_curve/{dataset}/false_negatives.csv"
                }
                
                TP = np.loadtxt(save_files['TP'])
                FP = np.loadtxt(save_files['FP'])
                TN = np.loadtxt(save_files['TN'])
                FN = np.loadtxt(save_files['FN'])

            tp, fp = len(TP), len(FP)
            tn, fn = len(TN), len(FN)

            all = ( tp + tn + fp + fn )
            TPR = ( tp ) / ( tp + fp ) * 100
            ACC = ( tp + tn ) / all * 100

            name_wrapper = self.name[10:50] + "..." if len(self.name) > 40 else self.name
            print(f"{name_wrapper} {dataset}".ljust(70), f"{f'{tp}'.ljust(7)} {f'{fp}'.ljust(7)} {f'{tn}'.ljust(7)} {f'{fn}'.ljust(7)}{f'{all}'.ljust(7)} -> Acc = {ACC:.2f}%, TPR = {TPR:.4f}%")

            return TP, FP, TN, FN

        # plot the ROC curve for a specific dataset (e.g. 'validation_data', 'real_background', etc.) over signal strength (VEM_charge)
        def ROC(self, dataset, **kwargs) -> None :

            TP, FP, TN, FN = self.load_and_print_performance(dataset, usecols = [0, 0, 0, 0])

            if kwargs.get("full_set", False):
                y, x = np.array(list(TP) + list(TN)), np.array(list(FP) + list(FN))
            else: y, x = TP, FP

            x = np.clip(x, a_min = 1e-6, a_max = None)
            y = np.clip(y, a_min = 1e-6, a_max = None)

            score_low, score_high = min([x.min(), y.min()]), max([x.max(), y.max()])
            last, current_x, current_y = score_low, 0, 0
            ROC_x, ROC_y = [],[]

            bins = np.geomspace(score_low, score_high, kwargs.get("n", 50))[::-1]
            norm = ( len(x) + len(y) ) * 5                  # why x5 ???

            for score_bin in bins:

                this_x = ((last >= x) & (x > score_bin)).sum()
                this_y = ((last >= y) & (y > score_bin)).sum()
                
                current_x += this_x / norm
                current_y += this_y / norm
                
                ROC_y.append(current_y), ROC_x.append(current_x)

                last = score_bin
            
            ROC_x.append(1), ROC_y.append(1)

            plt.title(kwargs.get("title",""))
            plt.xlim(-0.02,1.02)
            plt.ylim(-0.02,1.02)
            plt.rcParams.update({'font.size': 22})
            plt.xlabel("False positive rate"), plt.ylabel("True positive rate")
            plt.plot(ROC_x, ROC_y, label = kwargs.get("label", self.name + "/" + dataset), ls = kwargs.get("ls", "solid"), lw = 2)
            
            plt.plot([0,1],[0,1], ls = "--", c = "gray")

            tp, fp = len(TP), len(FP)
            tn, fn = len(TN), len(FN)

            all = ( tp + tn + fp + fn )
            TPR = ( tp ) / ( tp + fp ) * 100
            ACC = ( tp + tn ) / all * 100

            return ACC, TPR

        # precision and recall curve over signal strength (VEM_charge)
        def PRC(self, dataset : str, **kwargs) -> None :

            TP, FP, TN, FN = self.load_and_print_performance(dataset, usecols = [0, 0, 0, 0])

            TP = np.clip(TP, a_min = 1e-6, a_max = None)
            FP = np.clip(FP, a_min = 1e-6, a_max = None)
            FN = np.clip(FN, a_min = 1e-6, a_max = None)

            score_low = min([TP.min(), FP.min(), FN.min()])
            score_high = max([TP.max(), FP.max(), FN.max()])
            last = score_low
            PRC_x, PRC_y = [],[]

            for score_bin in np.geomspace(score_low, score_high, kwargs.get("n", 100))[::-1]:

                this_tp = ((last >= TP) & (TP > score_bin)).sum()
                this_fp = ((last >= FP) & (FP > score_bin)).sum()
                this_fn = ((last >= FN) & (FN > score_bin)).sum()
                last = score_bin

                if this_tp + this_fn != 0 and this_tp + this_fp != 0:
                    this_x = this_tp / (this_tp + this_fn)                                  # recall
                    this_y = this_tp / (this_tp + this_fp)                                  # precision

                else: continue

                # print(f"{last:.2e} -> {score_bin:.2e}: {this_x}, {this_y}")
                
                PRC_y.append(this_y), PRC_x.append(this_x)

            PRC_y.sort() # ???
            PRC_x.sort() # ???

            plt.xlim(-0.02,1.02)
            plt.ylim(0.48,1.02)
            plt.rcParams.update({'font.size': 22})
            plt.xlabel("Efficiency"), plt.ylabel("Precision")
            plt.plot(1 - np.array(PRC_x), PRC_y, c = kwargs.get("c", "steelblue"), label = self.name + "/" + dataset, ls = kwargs.get("ls", "solid"))
            plt.plot([0,1],[0.5,0.5,], ls = "--", c = "gray")

        # plot the classifiers efficiency at a given SPD and energy
        # TODO: Incorporate LDF / probability that station gets signal!!
        def spd_energy_efficiency(self, dataset : str, **kwargs) -> None : 

            TP, FP, TN, FN = self.load_and_print_performance(dataset, usecols = [(2, 3), 0, 0, (2, 3)])
            colormap = cmap.get_cmap("jet")

            TP_E, TP_SPD = TP[:, 0], TP[:, 1]
            FN_E, FN_SPD = FN[:, 0], FN[:, 1]

            fig = plt.figure()

            # Group predictions by energy first
            for e, energy in enumerate(EventGenerator.libraries.keys()):

                low, high = [float(x) for x in energy.split("_")]
                c = colormap(e / len(EventGenerator.libraries.keys()))
                x = np.where(np.logical_and( TP_E >= 10 ** low, TP_E < 10 ** high))
                o = np.where(np.logical_and( FN_E >= 10 ** low, FN_E < 10 ** high))

                x, o = TP_SPD[x], FN_SPD[o]
                bins = np.linspace(min(min(x), min(o)), max(max(x), max(o)), kwargs.get("n", 18))

                x_hist, sig = np.histogram(x, bins = bins)
                o_hist, sig = np.histogram(o, bins = bins)

                x_val, y_val = [], []

                for i, (x, o) in enumerate(zip(x_hist, o_hist)):

                    if x == 0 and o == 0:
                        continue
                    else:
                        distance = sig[i]
                        width = sig[i + 1] - sig[i]
                        height = 1/(x+o)**2 * np.sqrt(x * o**2 + o * x**2 + 2 * x*o)
                        elevation = (x / (x+o)) - height/2

                        x_val.append( 0.5 * (sig[1:] + sig[:-1])[i])
                        y_val.append( elevation + height/2 )

                        plt.gca().add_patch(Rectangle((distance, elevation), width, height, color = c, alpha = 0.6, lw = 0))

                plt.scatter(x_val,y_val, color = c, marker = "s")

            plt.ylim(0, 1.05)
            plt.xlabel("Shower plane distance / m")
            plt.ylabel("Trigger efficiency | Signal")

            bounds = [float(e.split("_")[0]) for e in EventGenerator.libraries.keys()] + [19.5]
            norm = BoundaryNorm(bounds, colormap.N)

            # create a second axes for the colorbar
            ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.8])
            # cb = ColorbarBase(ax2, cmap=colormap, norm=norm, label = r"log$_{10}(E)$")
            ColorbarBase(ax2, cmap=colormap, norm=norm, label = r"log$_{10}(E)$")

        @staticmethod
        def __header__() -> None : 

            global header_was_called
            header_was_called = True

            print("\nDATASET".ljust(72) + "TP      FP      TN      FN     sum")
    
    # Performance visualizers #######################################################################

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

            if logs.get("accuracy") >= 0.985:
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
            "one_layer_conv2d" : Architectures.__one_layer_conv2d__,
            "two_layer_conv2d" : Architectures.__two_layer_conv2d__,
            "minimal_conv2d" : Architectures.__minimal_conv2d__,
            "large_conv2d" : Architectures.__large_conv2d__
        }

    def __init__(self, name : str, set_architecture = None, supress_print : bool = False, **kwargs) -> None :

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
                choice = input(f"\nSelect epoch from {os.listdir('/cr/data01/filip/models/' + name)}\n Model: ")
                self.model = tf.keras.models.load_model("/cr/data01/filip/models/" + name + choice)
                self.epochs = int(choice[-1])
        else:

            self.model = tf.keras.Sequential()
            self.models[set_architecture](self.Architectures, self.model)
            self.epochs = 0

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

        self.save()

        # provide some metadata
        print("\nTraining exited normally. Onto providing metadata now...")

        true_positive_rate = make_dataset(self, ValidationSet, f"validation_data")

        with open(f"/cr/data01/filip/models/{self.name}/model_{self.epochs}/metrics.csv", "w") as metadata:
            for key, value in self.history.history.items():
                metadata.write(f"{key} {value[0]}\n")

            metadata.write(f"val_tpr {true_positive_rate}")

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

    def add(self, layer : str, **kwargs) -> None :
        print(self.layers[layer], layer, kwargs)
        self.layers[layer](**kwargs)

# Class for streamlined handling of multiple NNs with the same architecture
class Ensemble(NNClassifier):

    def __init__(self, name : str, set_architecture : str = None, n_models : int = GLOBAL.n_ensembles) -> None :

        r'''
        :name ``str``: specifies the name of the NN and directory in /cr/data01/filip/models/ENSEMBLES/
        :set_architecture ``str``: specifies the architecture (upon first initialization)

        :Keyword arguments:
        
        * *early_stopping_patience* (``int``) -- number of batches without improvement before training is stopped
        '''

        supress_print = False
        self.models = []

        for i in range(1, n_models + 1):
            ThisModel = NNClassifier("ENSEMBLES/" + name + f"/ensemble_{str(i).zfill(2)}/", set_architecture, supress_print)
            self.models.append(ThisModel)

            supress_print = True

        self.name = "ENSEMBLE_" + name

        print(f"{self.name}: {n_models} models successfully initiated\n")

    def train(self, Datasets : tuple, epochs : int, **kwargs) -> None:

        start = perf_counter_ns()

        for i, instance in enumerate(self.models,1):

            time_spent = (perf_counter_ns() - start) * 1e-9
            elapsed = strftime('%H:%M:%S', gmtime(time_spent))
            eta = strftime('%H:%M:%S', gmtime(time_spent * (len(self.models)/i - 1)))

            print(f"Model {i}/{len(self.models)}, {elapsed} elapsed, ETA = {eta}")

            instance.train(Datasets, epochs)

            Datasets[0].__reset__()
            Datasets[1].__reset__()

    def __call__(self, trace : np.ndarray) -> list :

        return [model(trace) for model in self.models]

    def load_and_print_performance(self, dataset : str) -> list : 

        # TP, FP, TN, FN 
        predictions = []

        # keep the iterable index in case of early breaking during debugging
        for i, model in enumerate(self.models):
            prediction = model.load_and_print_performance(dataset)
            predictions.append(prediction)

        return predictions

    def ROC(self, dataset : str, **kwargs : dict) -> None :

        warning_orange = '\033[93m'
        end_color_code = '\033[0m'

        predictions = self.load_and_print_performance(dataset)
        best_prediction, worst_prediction = None, None
        best_score, worst_score = -np.inf, np.inf
        all_predictions = [[] for i in range(2)]

        for i, prediction in enumerate(predictions): 

            TP, FP = prediction[0], prediction[1]

            # compare TPR across models
            current_score = ( len(TP) ) / ( len(TP) + len(FP) )

            if current_score > best_score:
                best_prediction = predictions[i]
                best_score = current_score
            if current_score < worst_score:
                worst_prediction = predictions[i]
                worst_score = current_score

            all_predictions[0] += list(TP)
            all_predictions[1] += list(FP)

        all_score = ( len(all_predictions[0]) ) / ( len(all_predictions[0]) + len(all_predictions[1]) )

        print(warning_orange + f"Overall performance (TPR): WORST = {worst_score * 100:.4f}%, BEST = {best_score * 100:.4f}%, ALL = {all_score * 100:.4f}%" + end_color_code)

        # draw the ROC curve for each of the (best, worst, avg.) predictions
        for i, prediction in enumerate([worst_prediction, best_prediction, all_predictions]):

            TP, FP = prediction[0], prediction[1]

            x = np.clip(FP, a_min = 1e-6, a_max = None)
            y = np.clip(TP, a_min = 1e-6, a_max = None)

            score_low, score_high = min([x.min(), y.min()]), max([x.max(), y.max()])
            last, current_x, current_y = score_low, 0, 0
            ROC_x, ROC_y = [],[]

            bins = np.geomspace(score_low, score_high, kwargs.get("n", 50))[::-1]
            norm = ( len(x) + len(y) ) * 5              # why x5 ???

            for score_bin in bins:

                this_x = ((last >= x) & (x > score_bin)).sum()
                this_y = ((last >= y) & (y > score_bin)).sum()
                
                current_x += this_x / norm
                current_y += this_y / norm
                
                ROC_y.append(current_y), ROC_x.append(current_x)

                last = score_bin
            
            ROC_x.append(1), ROC_y.append(1)

            plt.plot(ROC_x, ROC_y, lw = 2, alpha = 1 if i == 1 else 0.4, ls = "solid" if i == 1 else ":", c = kwargs.get("c", "steelblue"))

        plt.plot([],[], label = kwargs.get("label", f"{self.name} - {dataset}"), c = kwargs.get("c", "steelblue"))

        plt.xlim(-0.02,1.02)
        plt.ylim(-0.02,1.02)
        plt.plot([0,1],[0,1], ls = "--", c = "gray")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")

    def PRC(self, dataset : str) -> None :

        try:
            if header_was_called: pass

        except NameError:
            self.__header__()

        for i, model in enumerate(self.models):

            model.PRC(dataset, title = f"{self.name}", label = f"model instance {i}")

# Wrapper for currently employed station-level triggers (T1, T2, ToT, etc.)
# Information on magic numbers comes from Davids Mail on 10.03.22 @ 12:30pm
class HardwareClassifier(Classifier):

    def __init__(self, name : str = False) : 
        super().__init__(name or "HardwareClassifier")

    def __call__(self, trace : np.ndarray) -> bool : 
        
        # Threshold of 3.2 immediately gets promoted to T2
        # Threshold of 1.75 if a T3 has already been issued

        try:
            return self.Th(3.2, trace) or self.ToT(trace) or self.ToTd(trace) or self.MoPS(trace)
        except ValueError:
            return np.array([self.__call__(t) for t in trace])

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
    # note that this only ever gets applied to UB-like traces, with 25 ns binning
    def ToTd(self, signal : np.ndarray) -> bool : 

        # for information on this see GAP note 2018-01
        dt      = 25                                                                # UB bin width
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

        self.bin_centers = np.loadtxt("/cr/data01/filip/models/naive_bayes_classifier/bins.csv")
        self.signal = np.loadtxt("/cr/data01/filip/models/naive_bayes_classifier/signal.csv")
        self.background = np.loadtxt("/cr/data01/filip/models/naive_bayes_classifier/background.csv")
        self.quotient = self.signal / self.background
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