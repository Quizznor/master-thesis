from TriggerStudyBinaries.Classifier import TriggerClassifier, NNClassifier
from TriggerStudyBinaries.Signal import VEMTrace, Background, Baseline
from TriggerStudyBinaries.Generator import EventGenerator, Generator
import TriggerStudyBinaries.PerformanceTest as pt

_, AllEvents = EventGenerator("all", seed = True, real_background = True)

# CurrentTrigger = TriggerClassifier()
# pt.make_dataset(CurrentTrigger, AllEvents, "current_trigger_validation_data_real_background")

# MockLinearModel = NNClassifier("mock_model_all_linear/model_2")
# pt.make_dataset(MockLinearModel, AllEvents, "mock_linear_validation_data_real_background")

LargeModel = NNClassifier("large_model/model_5")
pt.make_dataset(LargeModel, AllEvents, "large_model_validation_data_real_background")

SmallConv2d = NNClassifier("small_conv2d/model_10")
pt.make_dataset(SmallConv2d, AllEvents, "small_model_validation_data_real_background")

CutFifthVEM = NNClassifier("minimal_conv2d_cut_0.2VEM/model_4")
pt.make_dataset(CutFifthVEM, AllEvents, "mock_fifth_validation_data_real_background_uncorrected")

CutHalfVEM = NNClassifier("minimal_conv2d_cut_0.5VEM/model_10")
pt.make_dataset(CutFifthVEM, AllEvents, "mock_half_validation_data_real_background_uncorrected")

CutOneVEM = NNClassifier("minimal_conv2d_cut_1.0VEM/model_2")
pt.make_dataset(CutOneVEM, AllEvents, "mock_one_validation_data_real_background_uncorrected")