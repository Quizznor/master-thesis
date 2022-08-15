from TriggerStudyBinaries_v4 import *

AllEvents = EventGenerator("all", ignore_low_vem = 4.63)

TestModel = NNClassifier("one_layer_conv2d")
TestModel.train(AllEvents, 10, "one_layer_conv2d_4.63VEM")