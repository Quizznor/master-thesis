from Binaries import *

AllEventsDownsampled = EventGenerator("all", split = 0, real_background = True, apply_downsampling = True)
# AllEventsDownsampled.unit_test(n_traces = 1000)

Hardware = HardwareClassifier()
NN_1_particle = NNClassifier("minimal_conv2d_1_particle")
NN_2_particle = NNClassifier("minimal_conv2d_2_particle")
make_dataset(Hardware, AllEventsDownsampled, "fullscale_downsampled")
make_dataset(NN_1_particle, AllEventsDownsampled, "fullscale_downsampled")
make_dataset(NN_2_particle, AllEventsDownsampled, "fullscale_downsampled")

AllEvents = EventGenerator("all", split = 0, real_background = True, apply_downsampling = False)
make_dataset(NN_1_particle, AllEventsDownsampled, "fullscale")
make_dataset(NN_2_particle, AllEventsDownsampled, "fullscale")