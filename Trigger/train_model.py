from TriggerStudyBinaries_v2.__configure__ import *

AllEvents = EventGenerator("all")
Models = Ensemble("one_layer_conv2d")

Models.train(AllEvents, 5)