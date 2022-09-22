from Binaries import *

AllEvents = EventGenerator("all", real_background = True)
# AllEvents[-1].unit_test()

# _ = input("\nPress ENTER to continue")

TestEnsemble = Ensemble("minimal_conv2d_stations_filtered", "one_layer_conv2d", 5)
TestEnsemble.train(AllEvents, 6)