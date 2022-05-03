# import numpy as np

# custom modules
from Classifiers import NNClassifier
from EventGenerators import EventGenerator
from PerformanceTest import test_noise_performance


def __setup__(cls : NNClassifier):

    def add(layer : str, **kwargs):
        cls.layers[layer](**kwargs)

    # define the input size
    input_size = 20000

    # define Network architecture
    add("Input", shape = (input_size, 1), batch_size = None)
    add("Conv1D", filters = 8, kernel_size = 11, strides = 5, activation = "relu")
    add("Conv1D", filters = 16, kernel_size = 11, strides = 10, activation = "relu")
    add("Conv1D", filters = 16, kernel_size = 11, strides = 10, activation = "relu")
    add("Conv1D", filters = 8, kernel_size = 11, strides = 5, activation = "relu")
    add("Flatten")
    add("Dense", units = 2, activation = "softmax")

# # Minimal model setup + training
# MinimalModel = NNClassifier("minimal_model/model_1")
# MinimalModel.train("01_simulation/component_signal/", 1)
# MinimalModel.save("minimal_model/")

# Minimal model performance
test_noise_performance("minimal_model/model_1", "02_simulation/component_signal/", "minimal_model_gen_1")
