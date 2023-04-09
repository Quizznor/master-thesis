from Binaries import *

start_time = perf_counter_ns()

# ALL ENERGIES
Events = EventGenerator("all", real_background = False)
NN = NNClassifier("test_all_energies", "two_layer_conv2d")

NN.train(Events, 10)

time_spent = (perf_counter_ns() - start_time) * 1e-9
elapsed = strftime('%H:%M:%S', gmtime(time_spent))

TP, _, _, FN = NN.load_and_print_performance("validation_data")

accuracy = len(TP) / (len(TP) + len(FN))

with open("/cr/data01/filip/models/test_all_energies/training_time.csv", "w") as file:
    file.write(elapsed + " " + str(accuracy))


# # HIGH ENERGIES
# Events = EventGenerator("19_19.5", real_background = True)
# NN = NNClassifier("test_high_energies", "two_layer_conv2d")

# NN.train(Events, 10)

# AllEvents = EventGenerator(":19_19.5", split = 1)
# ValFiles = EventGenerator(NN, real_background = True, split = 1)
# AllEvents.files += ValFiles.files
# NN.make_signal_dataset(AllEvents, "all_energies")

# time_spent = (perf_counter_ns() - start_time) * 1e-9
# elapsed = strftime('%H:%M:%S', gmtime(time_spent))

# TP, _, _, FN = NN.load_and_print_performance("all_energies")

# accuracy = len(TP) / (len(TP) + len(FN))

# with open("/cr/data01/filip/models/test_high_energies/training_time.csv", "w") as file:
#     file.write(elapsed + " " + str(accuracy))

