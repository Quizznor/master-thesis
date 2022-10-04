from Binaries import *

AllEvents = EventGenerator("all", real_background = True, ignore_low_vem = 1.0)
AllEvents[-1].unit_test()

_ = input("\nPress ENTER to continue")

TestEnsemble = Ensemble("minimal_conv2d_real_background_1.00VEM", "one_layer_conv2d", n_models = 10)

TestEnsemble.train(AllEvents, 3)