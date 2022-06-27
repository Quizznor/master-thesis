#!/usr/bin/python3

from binaries import *

if __name__ == "__main__":

    EventClassifier = Classifier()
    TrainingSet, ValidationSet = DataSetGenerator("01_simulation/signal/", train = True)
    EventClassifier.train(TrainingSet, ValidationSet, 1)
    EventClassifier.save("/cr/data01/filip/third_model/")
