from Binaries import *

"""
HardwareTriggers = HardwareClassifier()
HardwareTriggers.production_test(100000, apply_downsampling = True)

# perhaps this is caused by missing downsampling of trace?
# Also, what is the 'veto' after a trigger was issued?
"""

#######################################################################

Hardware = HardwareClassifier()
# Test1 = Ensemble("minimal_conv2d_real_background")
# Test2 = Ensemble("minimal_conv2d_real_background_1.00VEM")
# Test3 = Ensemble("minimal_conv2d_real_background_injections")
Test3 = NNClassifier("ENSEMBLES/minimal_conv2d_real_background_low_prior/ensemble_01")

Hardware.ROC("validation_data")

print("")
Test3.ROC("validation_data")

# print("")
# Test1.ROC("validation_data")

# print("")
# Test2.ROC("validation_data_no_cut")

# print("")
# Test3.ROC("validation_data")

# print("")
# Test1.ROC("validation_data_no_cut")