from TriggerStudyBinaries_v5 import *

# reseed the random number generator
def set_seed(seed : int) -> None : 
    np.random.seed(seed)
    random.seed(seed)

_, AllEvents = EventGenerator("all", seed = 42)

# CurrentTrigger = TriggerClassifier()
# make_dataset(CurrentTrigger, AllEvents, "current_trigger_validation_data")

# set_seed(42)

# MockLinearModel = NNClassifier("mock_model_all_linear/model_2")
# make_dataset(MockLinearModel, AllEvents, "one_layer_conv2d_validation_data")

# set_seed(42)

# CutFifthVEM = NNClassifier("minimal_conv2d_cut_0.20VEM/model_4")
# make_dataset(CutFifthVEM, AllEvents, "one_layer_conv2d_cut_0.20VEM_validation_data")

# set_seed(42)

# CutHalfVEM = NNClassifier("minimal_conv2d_cut_0.50VEM/model_10")
# make_dataset(CutHalfVEM, AllEvents, "one_layer_conv2d_cut_0.50VEM_validation_data")

# set_seed(42)

# CutOneVEM = NNClassifier("minimal_conv2d_cut_1.00VEM/model_2")
# make_dataset(CutOneVEM, AllEvents, "one_layer_conv2d_cut_1.00VEM_validation_data")

# set_seed(42)

# CutTwoVEM = NNClassifier("one_layer_conv2d_2.00VEM/ensemble_1/model_4/model_5")
# make_dataset(CutTwoVEM, AllEvents, "one_layer_conv2d_cut_2.00VEM_validation_data")

# set_seed(42)

# CutFiveVEM = NNClassifier("one_layer_conv2d_4.63VEM/model_converged")
# make_dataset(CutFiveVEM, AllEvents, "one_layer_conv2d_cut_4.63VEM_validation_data")

BayesClassifier = BayesianClassifier()
make_dataset(BayesClassifier, AllEvents, f"bayes_LHQ_validation_data")

# AllEvents.unit_test()