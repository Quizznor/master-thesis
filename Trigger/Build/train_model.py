from TriggerStudyBinaries_v7 import *

cuts = [0.2, 0.5, 1.0, 3.0]

for cut_value in cuts:
    AllEvents = EventGenerator("all", ignore_low_vem = cut_value, apply_downsampling = True)
    TestModel = NNClassifier(f"minimal_conv2d_{cut_value:.2f}VEM_downsampled")
    TestModel.train(AllEvents, 3)