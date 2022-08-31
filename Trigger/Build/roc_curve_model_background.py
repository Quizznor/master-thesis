from TriggerStudyBinaries_v7 import *

# reseed the random number generator
def set_seed(seed : int) -> None : 
    np.random.seed(seed)
    random.seed(seed)

_, AllEvents = EventGenerator("all", seed = 42, apply_downsampling = True)
# _, AllEventsDownsampled = EventGenerator("all", seed = 42, apply_downsampling = True)
# AllEvents.unit_test()

# CurrentTrigger = HardwareClassifier()
# make_dataset(CurrentTrigger, AllEvents, "validation_data_downsampled")

# set_seed(42)

CutNoVEM = NNClassifier("minimal_conv2d_cut_0.00VEM/model_converged")
make_dataset(CutNoVEM, AllEvents, "validation_data_downsampled")

set_seed(42)

CutFifthVEM = NNClassifier("minimal_conv2d_cut_0.20VEM/model_converged")
make_dataset(CutFifthVEM, AllEvents, "validation_data_downsampled")

set_seed(42)

CutHalfVEM = NNClassifier("minimal_conv2d_cut_0.50VEM/model_converged")
make_dataset(CutHalfVEM, AllEvents, "validation_data_downsampled")

set_seed(42)

CutOneVEM = NNClassifier("minimal_conv2d_cut_1.00VEM/model_converged")
make_dataset(CutOneVEM, AllEvents, "validation_data_downsampled")

set_seed(42)

# CutTwoVEM = NNClassifier("one_layer_conv2d_2.00VEM/ensemble_1/model_4/model_5")
# make_dataset(CutTwoVEM, AllEvents, "validation_data_downsampled")

# set_seed(42)

# CutFiveVEM = NNClassifier("one_layer_conv2d_4.63VEM/model_converged")
# make_dataset(CutFiveVEM, AllEvents, "validation_data_downsampled")

CutTenVEM = NNClassifier("minimal_conv2d_cut_10.0VEM/model_converged")
make_dataset(CutTenVEM, AllEvents, "validation_data_downsampled")

# BayesClassifier = BayesianClassifier()
# make_dataset(BayesClassifier, AllEvents, f"bayes_LHQ_validation_data_downsampled")

# AllEvents.unit_test()