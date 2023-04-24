from .__config__ import *
from .Signal import *
from .Generator import *
from .Classifier import *
from .Network import *

# Class for streamlined handling of multiple NNs with the same architecture
class Ensemble(NNClassifier):

    def __init__(self, name : str, set_architecture : str = None, n_models : int = GLOBAL.n_ensembles, **kwargs) -> None :

        r'''
        :name ``str``: specifies the name of the NN and directory in /cr/data01/filip/models/ENSEMBLES/
        :set_architecture ``str``: specifies the architecture (upon first initialization)

        :Keyword arguments:
        
        * *early_stopping_patience* (``int``) -- number of batches without improvement before training is stopped
        '''

        try:
            os.mkdir(f"/cr/data01/filip/models/ENSEMBLES/{name}")
        except FileExistsError: pass

        n_fail = 0

        supress_print = kwargs.get("supress_print", False)
        self.models = []

        for i in range(1, n_models + 1):

            try:
                ThisModel = NNClassifier("ENSEMBLES/" + name + f"/ensemble_{str(i).zfill(2)}/", set_architecture, supress_print)
                self.models.append(ThisModel)
            except FileNotFoundError:
                n_fail += 1

            supress_print = True

        self.name = "ENSEMBLE_" + name

        not kwargs.get("supress_print", False) and print(f"{self.name}: {n_models} models successfully initiated\n")
        if n_fail: print(f"{n_fail} models couldn't be initialized!")


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

    def save(self) -> None : 
        for model in self.models: model.save()

    def __call__(self, trace : np.ndarray) -> list :

        return [model(trace) for model in self.models]

    def load_and_print_performance(self, dataset : str, quiet = False) -> list : 

        # TP, FP, TN, FN 
        predictions = []

        # keep the iterable index in case of early breaking during debugging
        for i, model in enumerate(self.models):
            prediction = model.load_and_print_performance(dataset, quiet = quiet)
            predictions.append(prediction)

        return predictions

    def get_background_rates(self) -> tuple : 

        rate, rate_err = [], []

        for model in self.models:
            f,  df = np.loadtxt("/cr/data01/filip/models/" + model.name + "model_" + str(model.epochs) + "/production_test.csv", usecols = [0, 1])
            rate.append(f), rate_err.append(df)

        return np.array(rate), np.array(rate_err)
    
    def get_accuracy(self, dataset : str) -> tuple : 

        acc, acc_err = [], []

        for prediction in self.load_and_print_performance(dataset, quiet = True):
            TP, FP, TN, FN = prediction
            x, o = float(len(TP)), float(len(FN))
            accuracy = x / (x + o)
            err = 1/(x+o)**2 * np.sqrt( x**3 + o**3 - 2 * np.sqrt((x * o)**3) )

            acc.append(accuracy)
            acc_err.append(err)

        return np.array(acc), np.array(acc_err)

    def money_plot(self, dataset : str) -> None :

        rate, rate_err = self.get_background_rates()
        acc, acc_err = self.get_accuracy(dataset)
        x, y = np.mean(acc), np.mean(rate)

        if not os.path.isdir(f"/cr/users/filip/MoneyPlot/data/{self.name.split('/')[-1][9:]}"):
            os.makedirs(f"/cr/users/filip/MoneyPlot/data/{self.name.split('/')[-1][9:]}")

        save_matrix = np.dstack([acc, acc_err, rate, rate_err])[0]
        np.savetxt(f"/cr/users/filip/MoneyPlot/data/{self.name.split('/')[-1][9:]}/{dataset}.csv", save_matrix)


    def get_best_model(self, dataset : str) -> NNClassifier : 

        try:
            acc, acc_err, rate, rate_err = np.loadtxt(f"/cr/users/filip/MoneyPlot/data/{self.name[9:]}/{dataset}.csv", unpack = True)

            SN_ratios = rate / acc                              # this is technically a noise to signal ratio, hence we must MINIMIZE
            
            return self.models[np.argmin(SN_ratios)]

        except OSError:
            print(f"Crunching numbers for {self.name[9:]}: {dataset}...")
            self.money_plot(dataset)
            self.get_best_model(dataset)


    def make_signal_dataset(self, Dataset: "Generator", save_dir: str, n_showers: int = None, save_traces: bool = False) -> None:
        
        for model in self.models:
            model.make_signal_dataset(Dataset, save_dir, n_showers, save_traces)