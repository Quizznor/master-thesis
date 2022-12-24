from Binaries import *

AllEvents = EventGenerator("all", real_background = False, prior = 1e-5, ignore_particles = 1)
AllEvents[-1].unit_test()

# # _ = input("\nPress ENTER to continue")

# Network = NNClassifier("baseline_low_prior_1_particle", "one_layer_conv2d")
# Network.train(AllEvents, 10)
# Network2 = NNClassifier("minimal_conv2d_1_particle")
# Network3 = NNClassifier("minimal_conv2d_2_particle")
# Hardware = HardwareClassifier()

# AllEventsNoCut = EventGenerator("all", real_background = True)
# make_dataset(Network, AllEventsNoCut[-1], "validation_data_all_particles")

# Network2 = NNClassifier("minimal_conv2d_baseline")
# Hardware = HardwareClassifier()
# Hardware.ROC("validation_data")
# Network.ROC("validation_data")
# Network2.ROC("validation_data")
# Network2.ROC("validation_data_all_particles")
# Network3.ROC("validation_data")
# Network3.ROC("validation_data_all_particles")