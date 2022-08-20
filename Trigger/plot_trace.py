from TriggerStudyBinaries_v5 import *

Events = EventGenerator(["17_17.5"], split = 1, force_inject = 1)

traces, labels = Events.__getitem__(2, full_trace = True)

traces[0].__plot__()