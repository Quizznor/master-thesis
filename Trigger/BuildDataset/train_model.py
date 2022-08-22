from TriggerStudyBinaries_v6 import *

AllEvents = EventGenerator("all")

TestModel = NNClassifier("one_layer_conv2d")
TestModel.train(AllEvents, 10, "minimal_conv2d_cut_0.00VEM")