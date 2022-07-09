from TriggerStudyBinaries.PerformanceTest import TriggerProbabilityDistribution
from TriggerStudyBinaries.Classifier import TriggerClassifier, NNClassifier
from TriggerStudyBinaries.Generator import EventGenerator, Generator
from TriggerStudyBinaries.Signal import VEMTrace, Background

Classifier = NNClassifier(NNClassifier.__minimal_conv2d__)
Events = EventGenerator("all", split = 1, prior = 1)
TriggerProbabilityDistribution.build_dataset(Classifier, Events, "mock_model")