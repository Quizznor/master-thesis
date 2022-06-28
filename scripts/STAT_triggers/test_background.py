# builtin modules
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 22})

# custom modules
from binaries.Classifiers import NNClassifier, TriggerClassifier
from binaries.EventGenerators import EventGenerator
from binaries.Signal import VEMTrace

BackgroundData = EventGenerator("all", prior = 0, split = 1, sigma = 2, mu = [-2, 2])
Classifier = NNClassifier("/cr/data01/filip/all_traces_model/model_2")
False_Negatives, True_Negatives = 0, 0

for batch in range(BackgroundData.__len__()):
        traces, labels = BackgroundData.__getitem__(batch, for_training = False)
        print(f"Evaluating batch {batch}/{BackgroundData.__len__()}...", end = "\r")

        for Trace, _ in zip(traces, labels):
            if Classifier(Trace):
                False_Negatives += 1
            else: True_Negatives += 1

print(False_Negatives, True_Negatives, "                        ")
# outputs 0 66402 <- absolutely no false negatives! :o
# outputs 38650 27752 at mean and std = 2 ADC counts

# ExampleTrace = VEMTrace()
# t1, t2, t3 = ExampleTrace(pooling = False)

# ax1, ax2, ax3 = [plt.subplot2grid((3,7), (i,0), colspan = 6) for i in range(3)]
# ax4, ax5, ax6 = [plt.subplot2grid((3,7), (i,6), sharey = [ax1, ax2, ax3][i]) for i in range(3)]

# ax1.plot(range(len(t1)), t1)
# ax2.plot(range(len(t2)), t2)
# ax3.plot(range(len(t3)), t3)
# ax4.hist(t1, orientation = "horizontal", bins = 30, histtype = "step")
# ax5.hist(t2, orientation = "horizontal", bins = 30, histtype = "step")
# ax6.hist(t3, orientation = "horizontal", bins = 30, histtype = "step")

# for ax in [ax4, ax5, ax6]:
#     ax.axis("off")

# for ax in [ax1, ax2]:
#     ax.set_xticks([])

# ax3.set_xlabel("Time bin / 8.3 ns")
# ax1.set_ylabel("PMT #1 / VEM")
# ax2.set_ylabel("PMT #2 / VEM")
# ax3.set_ylabel("PMT #3 / VEM")

# plt.tight_layout()
# plt.subplots_adjust(wspace = 0)
# plt.show()
