from Binaries import *

AllEvents = EventGenerator("all", real_background = True)
# AllEvents[-1].unit_test()

# _ = input("\nPress ENTER to continue")

TestEnsemble = Ensemble("minimal_conv2d_real_background", "one_layer_conv2d", n_models = 8)

TestEnsemble.train(AllEvents, 3)