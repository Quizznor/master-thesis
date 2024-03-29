import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({'font.size': 22})

for file in os.listdir("/cr/data01/filip/first_simulation/test_second_model/"):

    std, label = file.split("_")
    std, label = float(std), label[:-4]
    confusion_matrix = np.loadtxt("/cr/data01/filip/first_simulation/test_second_model/" + file)
    accuracy = np.trace(confusion_matrix) / confusion_matrix.sum()

    marker = "o" if label == "network" else "s"
    color = "r" if label =="network" else "b"

    plt.scatter(std * 61.75, accuracy, marker = marker, c = color)

plt.scatter([],[], marker = "o", c = "r", label = "NN predicted")
plt.scatter([],[], marker = "s", c = "b", label = "T1/ToT predicted")
plt.xlabel("Baseline std / ADC counts")
plt.ylabel("Prediction accuracy")
plt.legend()
plt.show()