from TriggerStudyBinaries.Signal import VEMTrace, Background
from TriggerStudyBinaries.Generator import EventGenerator
from TriggerStudyBinaries.Classifier import NNClassifier

Real_Background = EventGenerator("all", prior = 0, split = 1, real_background = True, seed = True)
Model_Background = EventGenerator("all", prior = 0, split = 1, real_background = False, seed = True)

Classifier = NNClassifier("mock_model_all_linear/model_2")

for batch in range(Testingset_fake_background.__len__()):

    traces, labels = Testingset_fake_background.__getitem__(batch)
    print(f"Analysing batch {batch +1}/{Testingset_fake_background.__len__()}", end = "...\r")

    print(traces.shape)
