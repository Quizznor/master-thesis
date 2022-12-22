from Binaries import *

AllEvents = EventGenerator("all", real_background = True, ignore_particles = 1)
# AllEvents[-1].unit_test()

# # _ = input("\nPress ENTER to continue")

Network = NNClassifier("minimal_conv2d_1_particle", "one_layer_conv2d")
Network.train(AllEvents, 5)

Hardware = HardwareClassifier()

AllEventsNoCut = EventGenerator("all", real_background = True)
make_dataset(Network, AllEventsNoCut[-1], "validation_data_all_particles")

Hardware.ROC("validation_data")
Network.ROC("validation_data")
Network.ROC("validation_data_all_particles")