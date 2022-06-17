import numpy as np
import os, sys

source_path = "/cr/users/filip/data/second_simulation/tensorflow/signal/"
raw_files = np.unique([item[:13] for item in os.listdir(source_path)])

print(len(raw_files))