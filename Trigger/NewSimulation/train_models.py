from Binaries import *

# NNClassifier = NNClassifier("120_OneLayer_FullBandwidth_1Particle", "one_layer_conv2d")
Events = EventGenerator(["16_16.5"],  real_background = False, ignore_particle = 1)

Events[-1].physics_test(n_showers = 10000)
Events[-1].training_test(n_showers = 10000)

# NNClassifier.train(Events, 5)

