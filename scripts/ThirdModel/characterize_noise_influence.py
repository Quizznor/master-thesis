#!/usr/bin/python3

import sys

sys.dont_write_bytecode = True

from binaries import *
import time

def second_to_timestr(duration : float) -> str : 

    hours = int(duration // 3600)
    duration = duration % 3600
    minutes = int(duration // 60)
    seconds = int(duration % 60)

    return f"{hours}".zfill(2) + ":" + f"{minutes}".zfill(2) + ":" + f"{seconds}".zfill(2)

start = time.time()

EventClassifier = Classifier("/cr/data01/filip/third_model/model_1")

stds = np.unique([float(string[:string.rfind("_")]) for string in os.listdir("/cr/data01/filip/second_model/noise_studies")])

for step, std in enumerate(stds, 1):
    network_confusion_matrix = np.zeros(shape = (2,2))
    stepsize = 500

    PredictionDataset = DataSetGenerator("first_simulation/tensorflow/signal/", train = False, baseline_std = std)

    for i in range(PredictionDataset.__len__())[::stepsize]:
        traces, labels = PredictionDataset.__getitem__(i)
        iteration_counter = 0

        for trace, label in zip(traces, labels):
            predicted_network = EventClassifier.predict(trace = trace)

            # build confusion matrices
            if label[1] == 1:
                network_confusion_matrix[0][0] += 1 if predicted_network else 0
                network_confusion_matrix[0][1] += 1 if not predicted_network else 0
            elif label[0] == 1:
                network_confusion_matrix[1][0] += 1 if predicted_network else 0
                network_confusion_matrix[1][1] += 1 if not predicted_network else 0

            iteration_counter += 1
            progress = (i + iteration_counter)/(len(traces) * (PredictionDataset.__len__() / stepsize) )
            elapsed = time.time() - start

            print(f"Step {step}/{len(stds)} -> Progress: {progress : .2f}%, {second_to_timestr(elapsed)}, ETA = {second_to_timestr( (100/progress * elapsed) * (len(stds) - step) )}\r", end = "")            

    np.savetxt(f"/cr/data01/filip/third_model/noise_studies/{std}_network.txt", network_confusion_matrix)