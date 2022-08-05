from TriggerStudyBinaries_v2.__configure__ import *

AllEvents = EventGenerator("all", split = 1, prior = 0.5)

for batch in range(AllEvents.__len__()):

    traces, _ = AllEvents.__getitem__(batch, full_trace = True)

    for trace in traces:

        # for i in range(0, AllEvents.length, AllEvents.window_step):
        for i in AllEvents.__sliding_window__(trace):
            
            window, n_sig = trace.get_trace_window((i, i + AllEvents.window_length))
            print(window.shape, n_sig)

        raise StopIteration