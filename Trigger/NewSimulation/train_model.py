from Binaries import *

# AllEvents = EventGenerator("all", real_background = True, prior = 1e-5, ignore_particles = 1, ignore_vem = 1)
# AllEvents[-1].unit_test()

# # _ = input("\nPress ENTER to continue")

Network1 = NNClassifier("baseline_low_prior")
Network2 = NNClassifier("minimal_conv2d_1_particle")
Network3 = NNClassifier("minimal_conv2d_2_particle")
Network4 = NNClassifier("low_prior_1_particle")
Network5 = NNClassifier("particle_signal_cut_low_prior")

Hardware = HardwareClassifier()

# AllEventsNoCut = EventGenerator("all", real_background = True)
# make_dataset(Network4, AllEventsNoCut[-1], "no_edits_to_labels")

# Network5.train(AllEvents, 10)
# make_dataset(Network5, AllEventsNoCut[-1], "no_edits_to_labels")

Hardware.ROC("validation_data")
Network1.ROC("validation_data")
Network2.ROC("validation_data")
Network2.ROC("validation_data_all_particles")
Network3.ROC("validation_data")
Network3.ROC("validation_data_all_particles")
Network4.ROC("validation_data")
Network4.ROC("no_edits_to_labels")
Network5.ROC("validation_data")
Network5.ROC("no_edits_to_labels")