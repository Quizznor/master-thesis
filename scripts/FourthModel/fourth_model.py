import sys 

sys.dont_write_bytecode = True

from binaries import *

TrainingSet, ValidationSet = DataSetGenerator("01_simulation/signal/", train = True, pooling = True, baseline_std = 0.0570800219923027)
EventClassifier = Classifier("/cr/data01/filip/fourth_model/model_1")
EventClassifier.train(TrainingSet, ValidationSet, 2)
EventClassifier.save("/cr/data01/filip/fourth_model/")