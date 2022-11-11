from Binaries import *

# """
# HardwareTriggers = HardwareClassifier()
# HardwareTriggers.production_test(100000, apply_downsampling = True)

# # perhaps this is caused by missing downsampling of trace?
# # Also, what is the 'veto' after a trigger was issued?
# """

# #######################################################################

Hardware = HardwareClassifier()
Hardware.production_test(100000, apply_downsampling = True, station = "lo_qui_don")
# Test1 = Ensemble("minimal_conv2d_real_background")
# Test2 = Ensemble("minimal_conv2d_real_background_1.00VEM")
# Test5 = NNClassifier("minimal_conv2d_real_background_cut+prior")
# Test3 = Ensemble("minimal_conv2d_real_background_injections")
# Test4 = Ensemble("minimal_conv2d_real_background_low_prior")

# _, Events= EventGenerator("all", real_background = True, apply_downsampling = True)

# # make_dataset(Hardware, Events, "random_traces_downsampled")
# make_dataset(Test5, Events, "random_traces_downsampled")
# # make_dataset(Test1, Events, "random_traces_downsampled")
# # make_dataset(Test2, Events, "random_traces_downsampled")
# make_dataset(Test3, Events, "random_traces_downsampled")
# make_dataset(Test4, Events, "random_traces_downsampled")