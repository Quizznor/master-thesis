from Binaries import *

AllEvents = EventGenerator("all", real_background = True, prior = 1e-5, ignore_low_vem = 0.7)
AllEventsNoCut = EventGenerator("all", real_background = True, prior = 1e-5)
# AllEvents[-1].unit_test()

# _ = input("\nPress ENTER to continue")

TestEnsemble = NNClassifier("minimal_conv2d_real_background_cut+prior", "one_layer_conv2d")

TestEnsemble.train(AllEvents, 5)
make_dataset(TestEnsemble, AllEventsNoCut, "validation_data_no_cut")

Hardware = HardwareClassifier()

Hardware.ROC("validation_data")

print("")
TestEnsemble.ROC("validation_data_no_cut")