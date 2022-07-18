from TriggerStudyBinaries.Signal import VEMTrace, Background
from TriggerStudyBinaries.Generator import EventGenerator
from TriggerStudyBinaries.Classifier import NNClassifier

Testingset = EventGenerator("all", prior = 0, split = 1, force_inject = 1)
Classifier = NNClassifier("mock_model_all_linear/model_2")

false_positives = 0
n_total_traces = 0

for batch in range(Testingset.__len__()):

    traces, labels = Testingset.__getitem__(batch)
    print(f"Analysing batch {batch +1}/{Testingset.__len__()}", end = "...\r")

    n_total_traces += len(traces)

    for trace in traces:

        if Classifier(trace):
            false_positives += 1

print("\n", n_total_traces, false_positives)