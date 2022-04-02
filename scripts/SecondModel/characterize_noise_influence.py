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

BASELINE_NOISE_SPREAD = np.linspace(BASELINE_STD, 1, 10)
BASELINE_NOISE_SPREAD = np.linspace(BASELINE_NOISE_SPREAD[1],BASELINE_NOISE_SPREAD[2],10)
std = BASELINE_NOISE_SPREAD[int(sys.argv[1])]
start = time.time()

EventClassifier = Classifier("/cr/data01/filip/second_simulation/tensorflow/model/model_1")

legacy_confusion_matrix = np.zeros(shape = (2,2))
network_confusion_matrix = np.zeros(shape = (2,2))
stepsize = 100

PredictionDataset = DataSetGenerator("second_simulation/tensorflow/signal/", train = False, baseline_std = std)

for i in range(PredictionDataset.__len__())[::stepsize]:
    traces, labels = PredictionDataset.__getitem__(i)

    for trace, label in zip(traces, labels):
        predicted_legacy = VEMTrace(label, trace = trace).has_triggered()
        predicted_network = EventClassifier.predict(trace = trace)

        # build confusion matrices
        if label[1] == 1:
            legacy_confusion_matrix[0][0] += 1 if predicted_legacy else 0
            legacy_confusion_matrix[0][1] += 1 if not predicted_legacy else 0
            network_confusion_matrix[0][0] += 1 if predicted_network else 0
            network_confusion_matrix[0][1] += 1 if not predicted_network else 0
        elif label[0] == 1:
            legacy_confusion_matrix[1][0] += 1 if predicted_legacy else 0
            legacy_confusion_matrix[1][1] += 1 if not predicted_legacy else 0
            network_confusion_matrix[1][0] += 1 if predicted_network else 0
            network_confusion_matrix[1][1] += 1 if not predicted_network else 0

        progress = ((i+1)/(PredictionDataset.__len__() / stepsize))
        elapsed = time.time() - start

        print(f"Progress: {progress : .2f}%, {second_to_timestr(elapsed)}, ETA = {second_to_timestr( 100/progress * elapsed )}\r", end = "")            

np.savetxt(f"/cr/data01/filip/second_simulation/noise_studies/{std}_legacy.txt", legacy_confusion_matrix)
np.savetxt(f"/cr/data01/filip/second_simulation/noise_studies/{std}_network.txt", network_confusion_matrix)