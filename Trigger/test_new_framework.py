from TriggerStudyBinaries.Signal import VEMTrace, Background
from TriggerStudyBinaries.Generator import EventGenerator
from TriggerStudyBinaries.Classifier import NNClassifier

Trainingset, Testingset = EventGenerator("all", prior = 0.5)
# Classifier = NNClassifier("mock_model_all/model_1")

# Classifier.train((Trainingset, Testingset), epochs = 5, verbose = 1)
# Classifier.save("mock_model_all/")

traces, labels = Trainingset.__getitem__(0)