from Binaries import *

"""
HardwareTriggers = HardwareClassifier()
HardwareTriggers.production_test(100000, apply_downsampling = True)

# perhaps this is caused by missing downsampling of trace?
# Also, what is the 'veto' after a trigger was issued?
"""

#######################################################################

Real_Background_NN = NNClassifier("minimal_conv2d_real_background")
Real_Background_NN.production_test(100000, apply_downsampling = True)