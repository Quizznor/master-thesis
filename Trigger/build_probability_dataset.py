from TriggerStudyBinaries.PerformanceTest import TriggerProbabilityDistribution
from TriggerStudyBinaries.Classifier import TriggerClassifier, NNClassifier
from TriggerStudyBinaries.Generator import EventGenerator, Generator
from TriggerStudyBinaries.Signal import VEMTrace, Background

Testingset = EventGenerator("all", prior = 1, split = 1, seed = True, mu = [0,0], sigma = 0)
Classifier = TriggerClassifier()

# Trainingset, Testingset = EventGenerator("all", prior = 1, seed = True)
# Classifier = NNClassifier("mock_model_two_layers/model_2")

TriggerProbabilityDistribution.build_dataset(Classifier, Testingset, "current_trigger_zero_baseline.txt")