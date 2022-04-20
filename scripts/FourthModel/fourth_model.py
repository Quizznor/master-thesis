import sys 

sys.dont_write_bytecode = True

from binaries import *

TrainingSet, ValidationSet = DataSetGenerator("01_simulation/signal/", train = True, pooling = True)
EventClassifier = Classifier()
EventClassifier.train(TrainingSet, ValidationSet, 1)
EventClassifier.save("/cr/data01/filip/fourth_model/")