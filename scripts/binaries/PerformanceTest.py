from Classifiers import TriggerClassifier, NNClassifier
from EventGenerators import Generator
from Signal import VEMTrace
import numpy as np
import typing
import time

def lateral_trigger_probability(Classifier : typing.Any, Dataset : Generator) -> tuple :

    hits, misses = [], []

    for batch in range(Dataset.__len__()):
        traces, labels = Dataset.__getitem__(batch, for_training = False)

        for Trace, label in zip(traces, labels):
            predicted_label = Classifier(Trace)

            if predicted_label == label:
                hits.append(Trace._spdistance)
            else:
                misses.append(Trace._spdistance)
    
    return hits, misses

def energy_trigger_probability(Classifier : typing.Any, Dataset : Generator) -> tuple :

    hits, misses = [], []

    # TODO link energy to prediction

    for batch in range(Dataset.__len__()):
        event_name = Dataset.__files[batch]
        traces, labels = Dataset.__getitem__(batch, for_training = False)

        # for Trace, label in zip(traces, labels):
        #     predicted_label = Classifier(Trace)

        #     if predicted_label == label:
        #         hits.append(Trace._spdistance)
        #     else:
        #         misses.append(Trace._spdistance)
    
    return hits, misses

# this uses outdated Signal / EventGenerator class structure and is obsolete
'''
def test_noise_performance(network_dir : str, dataset_dir : str, save_dir : str, noise_levels : list = None) -> typing.NoReturn:

    def second_to_timestr(duration : float) -> str : 

        hours = int(duration // 3600)
        duration = duration % 3600
        minutes = int(duration // 60)
        seconds = int(duration % 60)

        return f"{hours}".zfill(2) + ":" + f"{minutes}".zfill(2) + ":" + f"{seconds}".zfill(2)

    start = time.time()

    DEFAULT_NOISE = [0.00809717, 0.02034288, 0.03258859, 0.04483431, 0.05708002, 0.06932574, 0.08157145, 0.09381716, 0.10606288, 0.11830859, 0.13055431, 0.14280002]
    noise_levels = DEFAULT_NOISE if noise_levels is None else noise_levels
    Dataset = EventGenerator(dataset_dir, train = False)
    Network = NNClassifier(network_dir)

    total_steps = Dataset.__len__() * len(noise_levels)

    for j, noise in enumerate(noise_levels):

        Dataset = EventGenerator(dataset_dir, baseline_std = noise, train = False)
        network_confusion_matrix = np.zeros(shape = (2,2))

        # skim over ~1% of the dataset:
        for i in range(Dataset.__len__())[::100]:

            current_steps = j * Dataset.__len__() + i + 1
            traces, labels = Dataset.__getitem__(i)
            

            for trace, label in zip(traces, labels):
                predicted_network = Network.predict(trace = trace)

                # build confusion matrices
                if label[1] == 1:
                    network_confusion_matrix[0][0] += 1 if predicted_network else 0
                    network_confusion_matrix[0][1] += 1 if not predicted_network else 0
                elif label[0] == 1:
                    network_confusion_matrix[1][0] += 1 if predicted_network else 0
                    network_confusion_matrix[1][1] += 1 if not predicted_network else 0

            elapsed = time.time() - start
            print(f"Looping through traces {current_steps}/{total_steps}   ETA = {second_to_timestr((total_steps/current_steps-1) * elapsed)}", end = "\r")

        np.savetxt(f"/cr/data01/filip/noise_studies/{save_dir}/{noise}_network.txt", network_confusion_matrix)
'''

if __name__ == "__main__":

    Classifier = NNClassifier("/cr/data01/filip/minimal_model/model_2")