import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({'font.size': 22})

for file in os.listdir("/cr/data01/filip/second_model/noise_studies"):

    std, label = file.split("_")
    std, label = float(std), label[:-4]
    confusion_matrix = np.loadtxt("/cr/data01/filip/second_model/noise_studies/" + file)
    accuracy = np.trace(confusion_matrix) / confusion_matrix.sum()

    marker = "o" if label == "network" else "_"
    color = "r" if label =="network" else "b"
    linestyle = "--" if label == "network" else None

    plt.scatter(std * 61.75, accuracy, marker = marker, c = color)
    # plt.errorbar(std * 61.75, accuracy, marker = marker, c = color, ls = linestyle, yerr = 1/np.sqrt(confusion_matrix.sum()))

    # if label == "network":
        # plt.plot(std * 61.75, accuracy, ls = "--", c = color)

for file in os.listdir("/cr/data01/filip/third_model/noise_studies"):

    std, label = file.split("_")
    std, label = float(std), label[:-4]
    confusion_matrix = np.loadtxt("/cr/data01/filip/third_model/noise_studies/" + file)
    accuracy = np.trace(confusion_matrix) / confusion_matrix.sum()

    plt.scatter(std * 61.75, accuracy, marker = "o", c = "green")
    # plt.errorbar(std * 61.75, accuracy, marker = "o", c = "green", ls = "--", yerr = 1/np.sqrt(confusion_matrix.sum()))
    # plt.plot(std * 61.75, accuracy, ls = "--", c = "green")

for file in os.listdir("/cr/data01/filip/fourth_model/noise_studies_gen_1"):

    std, label = file.split("_")
    std, label = float(std), label[:-4]
    confusion_matrix = np.loadtxt("/cr/data01/filip/fourth_model/noise_studies_gen_1/" + file)
    accuracy = np.trace(confusion_matrix) / confusion_matrix.sum()

    plt.scatter(std * 61.75, accuracy, marker = "s", c = "k")
    # plt.errorbar(std * 61.75, accuracy, marker = "s", c = "k", ls = "-", yerr = 1/np.sqrt(confusion_matrix.sum()))
    # plt.plot(std * 61.75, accuracy, ls = "--", c = "k")

for file in os.listdir("/cr/data01/filip/fourth_model/noise_studies_gen_2"):

    std, label = file.split("_")
    std, label = float(std), label[:-4]
    confusion_matrix = np.loadtxt("/cr/data01/filip/fourth_model/noise_studies_gen_2/" + file)
    accuracy = np.trace(confusion_matrix) / confusion_matrix.sum()

    plt.scatter(std * 61.75, accuracy, marker = "s", c = "orange")
    # plt.errorbar(std * 61.75, accuracy, marker = "s", c = "green", ls = "--", yerr = 1/np.sqrt(confusion_matrix.sum()))
    # plt.plot(std * 61.75, accuracy, ls = "--", c = "orange")

for file in os.listdir("/cr/data01/filip/noise_studies/minimal_model/"):

    std, label = file.split("_")
    std, label = float(std), label[:-4]
    confusion_matrix = np.loadtxt("/cr/data01/filip/noise_studies/minimal_model/" + file)
    accuracy = np.trace(confusion_matrix) / confusion_matrix.sum()

    plt.scatter(std * 61.75, accuracy, marker = "x", c = "green")
    # plt.errorbar(std * 61.75, accuracy, marker = "s", c = "green", ls = "--", yerr = 1/np.sqrt(confusion_matrix.sum()))

for file in os.listdir("/cr/data01/filip/noise_studies/minimal_model/"):

    std, label = file.split("_")
    std, label = float(std), label[:-4]
    confusion_matrix = np.loadtxt("/cr/data01/filip/noise_studies/minimal_model_gen_1/" + file)
    accuracy = np.trace(confusion_matrix) / confusion_matrix.sum()

    plt.scatter(std * 61.75, accuracy, marker = "x", c = "red")
    # plt.errorbar(std * 61.75, accuracy, marker = "s", c = "green", ls = "--", yerr = 1/np.sqrt(confusion_matrix.sum()))

plt.scatter([],[], marker = "o", c = "r", label = "Sequential model")
plt.scatter([],[], marker = "o", c = "g", label = "Sequential + pooling")
plt.scatter([],[], marker = "_", c = "b", label = "T1/ToT predicted")
plt.scatter([],[], marker = "s", c = "k", label = "Convolution + pooling Gen 1")
plt.scatter([],[], marker = "s", c = "orange", label = "Convolution + pooling Gen 2")
plt.scatter([],[], marker = "x", c = "green", label = "Minimal convolution network")
plt.scatter([],[], marker = "x", c = "red", label = "Minimal convolution network Gen 1")
plt.xlabel("Baseline std / ADC counts")
plt.ylabel("Prediction accuracy")
plt.legend()
plt.show()