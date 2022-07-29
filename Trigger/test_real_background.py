from TriggerStudyBinaries.Classifier import TriggerClassifier, NNClassifier
from TriggerStudyBinaries.Signal import VEMTrace, Background, Baseline
from TriggerStudyBinaries.Generator import EventGenerator, Generator
import TriggerStudyBinaries.PerformanceTest as pt
import matplotlib.pyplot as plt

_, AllEventsModelBackground = EventGenerator("all", seed = True, real_background = False)
_, AllEventsModelBackground = EventGenerator("all", seed = True, real_background = True)

MinimalConv2dModel = NNClassifier("small_conv2d/model_10")
pt.make_dataset(MinimalConv2dModel, AllEventsModelBackground, "small_conv2d_model_background")
pt.make_dataset(MinimalConv2dModel, AllEventsRealBackground, "small_conv2d_real_background")