# custom modules
from binaries.Classifiers import NNClassifier
from binaries.EventGenerators import EventGenerator


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
    add("Output", units = 2, activation = "softmax")

# # Minimal model setup + training
MinimalModel = NNClassifier("/cr/data01/filip/all_traces_model/model_1")
DataGenerator = EventGenerator("all")
MinimalModel.train(DataGenerator, 1)
MinimalModel.save("all_traces_model/")
