#!/usr/bin/python3

import numpy as np

np.savetxt("/cr/users/filip/condor_output/accumulated_signal.csv", np.zeros((600, 2048)))
np.savetxt("/cr/users/filip/condor_output/accumulated_background.csv", np.zeros((100, 2048)))