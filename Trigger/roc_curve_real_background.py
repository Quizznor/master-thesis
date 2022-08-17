from TriggerStudyBinaries_v4 import *

# reseed the random number generator
def set_seed(seed : int) -> None : 
    np.random.seed(seed)
    random.seed(seed)

_, AllEvents = EventGenerator("all", seed = 42, real_background = True, force_inject = 0)

# CurrentTrigger = TriggerClassifier()
# make_dataset(CurrentTrigger, AllEvents, "current_trigger_random_traces")

# set_seed(42)

# MockLinearModel = NNClassifier("mock_model_all_linear/model_2")
# make_dataset(MockLinearModel, AllEvents, "one_layer_conv2d_random_traces")

# set_seed(42)

# CutFifthVEM = NNClassifier("minimal_conv2d_cut_0.20VEM/model_4")
# make_dataset(CutFifthVEM, AllEvents, "one_layer_conv2d_cut_0.20VEM_random_traces")

# set_seed(42)

# CutHalfVEM = NNClassifier("minimal_conv2d_cut_0.50VEM/model_10")
# make_dataset(CutHalfVEM, AllEvents, "one_layer_conv2d_cut_0.50VEM_random_traces")

# set_seed(42)

# CutOneVEM = NNClassifier("minimal_conv2d_cut_1.00VEM/model_2")
# make_dataset(CutOneVEM, AllEvents, "one_layer_conv2d_cut_1.00VEM_random_traces")

# set_seed(42)

# CutTwoVEM = NNClassifier("one_layer_conv2d_2.00VEM/ensemble_1/model_4/model_5")
# make_dataset(CutTwoVEM, AllEvents, "one_layer_conv2d_cut_2.00VEM_random_traces")

set_seed(42)

CutFiveVEM = NNClassifier("one_layer_conv2d_4.63VEM/model_converged")
make_dataset(CutFiveVEM, AllEvents, "one_layer_conv2d_cut_4.63VEM_random_traces")

AllEvents.unit_test()