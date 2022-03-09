import sys, os
sys.dont_write_bytecode = True

# insert at the beginning, to avoid naming conflicts
sys.path.insert(1, "/cr/users/filip/scripts/FirstModel")
from first_model import VEMTrace, TraceGenerator
from second_model import Trigger

DataGenerator = TraceGenerator(train = True, split = 1, input_shape = 2000, fix_seed = True, shuffle = False, verbose = False)

# have 100 signal files
for i in range(100):

    traces, labels = DataGenerator.__getitem__()