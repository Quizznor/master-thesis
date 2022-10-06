from Binaries import *

AllEvents = EventGenerator("all", real_background = True, prior = 1e-5)
AllEvents[-1].unit_test()

_ = input("\nPress ENTER to continue")

TestEnsemble = Ensemble("minimal_conv2d_real_background_low_prior", "one_layer_conv2d", n_models = 10)

TestEnsemble.train(AllEvents, 5)