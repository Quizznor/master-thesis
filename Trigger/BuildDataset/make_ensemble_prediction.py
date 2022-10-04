from Binaries import *

_, Dataset = EventGenerator("all", real_background = True)
Dataset.unit_test()

RealBackgroundNN = Ensemble("minimal_conv2d_real_background_1.00VEM")
make_dataset(RealBackgroundNN, Dataset, "validation_data_no_cut")