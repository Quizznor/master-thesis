from TriggerStudyBinaries.PerformanceTest import TriggerProbabilityDistribution
from TriggerStudyBinaries.Classifier import TriggerClassifier, NNClassifier
from TriggerStudyBinaries.Generator import EventGenerator, Generator
from TriggerStudyBinaries.Signal import VEMTrace, Background

# Classifier = NNClassifier("mock_model_all/model_3")
# Trainingset, Testingset = EventGenerator("all", prior = 1)
# TriggerProbabilityDistribution.build_dataset(Classifier, Testingset, "mock_model_val_3")

TriggerProbabilityDistribution.profile_plot("/cr/data01/filip/ROC_curves/mock_model_val_3.txt")