from Binaries import *

Events = EventGenerator("all", real_background = True, force_inject = 0)
# Events.unit_test()

NetworkNoCut = NNClassifier("minimal_conv2d_cut_0.00VEM", supress_print = True)
make_dataset(NetworkNoCut, Events[-1], "real_background_corrected")

NetworkFifthCut = NNClassifier("minimal_conv2d_cut_0.20VEM", supress_print = True)
make_dataset(NetworkFifthCut, Events[-1], "real_background_corrected")

# Events = EventGenerator("all", real_background = True, force_inject = 0)

NetworkHalfCut = NNClassifier("minimal_conv2d_cut_0.50VEM", supress_print = True)
make_dataset(NetworkHalfCut, Events[-1], "real_background_corrected")

# Events = EventGenerator("all", real_background = True, force_inject = 0)

NetworkOneCut = NNClassifier("minimal_conv2d_cut_1.00VEM", supress_print = True)
make_dataset(NetworkOneCut, Events[-1], "real_background_corrected")


