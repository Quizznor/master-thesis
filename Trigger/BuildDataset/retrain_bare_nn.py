from Binaries import *

# AllEvents = EventGenerator("all", real_background = True)
AllEventsDownsampled = EventGenerator("all", real_background = True, apply_downsampling = True)
# # AllEvents[-1].unit_test()

# TestEnsemble = Ensemble("minimal_conv2d_real_background", "one_layer_conv2d")
# TestEnsemble.train(AllEvents, 5)

TestEnsemble = Ensemble("minimal_conv2d_real_background")
make_dataset(TestEnsemble, AllEventsDownsampled[-1], "random_traces_downsampled")