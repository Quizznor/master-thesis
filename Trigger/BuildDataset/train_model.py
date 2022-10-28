from Binaries import *

AllEvents = EventGenerator("all", real_background = True, force_inject = 1) # prior = 1e-5)  # ignore_low_vem = 1.0)
AllEventsNoCut = EventGenerator("all", real_background = True) #, prior = 1e-5) #
# AllEvents[-1].unit_test()

# # _ = input("\nPress ENTER to continue")

TestEnsemble = Ensemble("minimal_conv2d_real_background_injections", "one_layer_conv2d")
TestEnsemble.train(AllEvents, 5)

make_dataset(TestEnsemble, AllEventsNoCut[-1], "validation_data_no_injections")

Hardware = HardwareClassifier()

Hardware.ROC("validation_data")
TestEnsemble.ROC("validation_data")
TestEnsemble.ROC("validation_data_no_injections")