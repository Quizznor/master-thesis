import sys 

sys.dont_write_bytecode = True

from binaries import *

one_up_baseline = 0.0570800219923027

TrainingSet, ValidationSet = DataSetGenerator("01_simulation/component_signal/", train = True, pooling = True, baseline_std = one_up_baseline)
EventClassifier = Classifier("/cr/data01/filip/fifth_model/model_1")
EventClassifier.train(TrainingSet, ValidationSet, 2)
EventClassifier.save("/cr/data01/filip/fifth_model/")