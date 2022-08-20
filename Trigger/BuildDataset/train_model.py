from TriggerStudyBinaries_v5 import *

AllEvents = EventGenerator("all")

TestModel = NNClassifier("large_conv2d")
TestModel.train(AllEvents, 10, "large_conv2d")