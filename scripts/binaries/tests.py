from Classifiers import NNClassifier

def __setup__(cls : NNClassifier):

    def add(layer : str, **kwargs):
        cls.layers[layer](**kwargs)

    # define the input size
    input_size = 20000

    # define Network architecture
    add("Input", shape = (input_size, 1), batch_size = None)
    add("Conv1D", filters = 8, kernel_size = 11, strides = 5, activation = "relu")
    add("Conv1D", filters = 16, kernel_size = 11, strides = 10, activation = "relu")
    add("Conv1D", filters = 32, kernel_size = 11, strides = 10, activation = "relu")
    add("Flatten")
    add("Dense", units = 2, activation = "softmax")

Test = NNClassifier(__setup__)
Test.train("02_simulation/component_signal/", epochs = 1)