#!/usr/bin/python3

import sys, os
import numpy as np

class VEMTrace:

    def __init__(self, trace_length : int, mu : float = 0, std : float = 0.1) -> None :

        self.length = trace_length
        self.__pmt_1 = np.random.normal(mu, std, trace_length)
        self.__pmt_2 = np.random.normal(mu, std, trace_length)
        self.__pmt_3 = np.random.normal(mu, std, trace_length)

    def __iadd__(self, signal):

        assert self.length > len(signal), "SIGNAL DOES NOT FIT INTO BASELINE!\n"

        start = np.random.randint(-self.length, -len(signal))
        self.__pmt_1[start : start + len(signal)] += signal
        self.__pmt_2[start : start + len(signal)] += signal
        self.__pmt_3[start : start + len(signal)] += signal

        return self

    def get(self):

        return self.__pmt_1, self.__pmt_2, self.__pmt_3 

class TraceGenerator():

    def __init__(self, train : bool, split : float, input_shape : int, fix_seed : bool = False, shuffle : bool = True, verbose : bool = True ) -> None:

        assert 0 <= split <= 1, "PLEASE PROVIDE A VALID SPLIT: 0 < split < 1"

        self.train = train              # wether object is used for training or validation
        self.split = int(100 * split)   # 0 - 100, percentage of data for this generator
        self.input_shape = input_shape  # input length of the data, trace duration = 8.3 ns * length
        self.fix_seed = fix_seed        # whether or not RNG seed is fixed, for reproducibility
        self.shuffle = shuffle          # whether or not to shuffle signal files at the end of each epoch
        self.verbose = verbose          # whether or not to print output to stdout during training

        self.n_events = [0, 0]          # number of background/signal traces respectively
        self.__file_count = 0           # number of calls to __getitem__ per epoch
        self.__epochs = 0               # number of epochs that the generator was used for
        
        self.__working_directory = "/cr/users/filip/data/first_simulation/tensorflow/signal/"

        if self.train:
            self.__signal_files = os.listdir(self.__working_directory)[:self.split]
        elif not self.train:
            self.__signal_files = os.listdir(self.__working_directory)[-self.split:]

    # generator that creates one batch, loads one signal file (~2000 events)
    # and randomly shuffles background events inbetween the signal traces
    def __getitem__(self, index) -> tuple :

        self.__traces, self.__labels = [], []
        self.__file_count += 1

        # generate mixed set of signals and baselines
        events = np.loadtxt(self.__working_directory + self.__signal_files[index])

        for signal in events:

            # decide whether baseline gets interlaced
            choice = np.random.choice([0,1])

            while not choice:
                self.add_event(choice, signal)      # add background event to list
                choice = np.random.choice([0,1])    # add another background event?
            self.add_event(choice, signal)          # add signal event to list

        self.verbose and print("[" + self.__file_count * "-" + (self.__len__() - self.__file_count) * " " + f"] {self.__file_count}/{self.__len__()}" )

        # true batch size is not EXACTLY batch size, this should be okay
        return (np.array(self.__traces), np.array(self.__labels))

    # split into validation set and training set repectively
    def __len__(self):
        return self.split

    # add labelled trace consisting of either just baseline or baseline + signal             
    def add_event(self, choice : int, signal : np.ndarray) -> tuple:

        self.n_events[choice] += 1
        Baseline = VEMTrace(self.input_shape)
        if choice == 1: Baseline += signal

        choice_encoded = [0,1] if choice else [1,0]
        self.__traces.append(Baseline.get())
        self.__labels.append(choice_encoded)

    # called by model.fit at the end of each epoch
    def on_epoch_end(self) -> None : 

        self.verbose and print(f"\nEPOCH {str(self.__epochs).zfill(3)} DONE; {self.n_events} events in buffer " + 100 * "_" + "\n")

        self.shuffle and np.random.shuffle(self.__signal_files)
        self.__file_count = 0
        self.n_events = [0, 0]
        self.__epochs += 1

    # fix random number generator seed for reproducibility
    @staticmethod
    def set_random_number_generator_seed(seed : int = 0) -> None :
        np.random.seed(seed)

class Trigger():

    def __init__(self, trace : tuple) -> None :

        self.length = len(trace[0])
        self.__pmt_1 = trace[0]
        self.__pmt_2 = trace[1]
        self.__pmt_3 = trace[2]

    # T1 threshold = 1.75 VEM in all 3 PMTs
    def T1_trigger(self):
        return self.absolute_threshold_trigger(1.75)

    # T2 threshold = 3.20 VEM in all 3 PMTs
    def T2_trigger(self):
        return self.absolute_threshold_trigger(3.20)

    # requires 2/3 PMTs have >= 13 bins over 0.2 VEM within ~ 1 μs
    # TODO is this performant enough? I trimmed it down as good as I can
    def ToT_trigger(self, window_length : int = 120, threshold : float = 0.2) -> bool :

        # count initial active bins
        pmt1_active = len(self.__pmt_1[:window_length][self.__pmt_1[:window_length] > threshold])
        pmt2_active = len(self.__pmt_2[:window_length][self.__pmt_2[:window_length] > threshold])
        pmt3_active = len(self.__pmt_3[:window_length][self.__pmt_3[:window_length] > threshold])

        for i in range(window_length,self.length):

            # check if ToT conditions are met
            ToT_trigger = [pmt1_active >= 13, pmt2_active >= 13, pmt3_active >= 13]

            if ToT_trigger.count(True) >= 2:
                return True

            # overwrite oldest bin and reevaluate
            pmt1_active += self.updated_bin_count(i, self.__pmt_1)
            pmt2_active += self.updated_bin_count(i, self.__pmt_2)
            pmt3_active += self.updated_bin_count(i, self.__pmt_3)

        return False

    # Helper for ToT trigger
    @staticmethod
    def updated_bin_count(index : int, array: np.ndarray, window_length : int = 120, threshold : float = 0.2) -> int :

        # is new bin active?
        if array[index] >= threshold:
            updated_bin_count = 1
        else:
            updated_bin_count = 0

        # was old bin active?
        if array[index - window_length] >= threshold:
            updated_bin_count -= 1

        return updated_bin_count

    # Check for coincident signals in all three photomultipliers
    def absolute_threshold_trigger(self, threshold : float) -> bool :

        # hierarchy doesn't (shouldn't?) matter, since we need coincident signal anyway
        for i in range(self.length):

            if self.__pmt_1[i] >= threshold:
                if self.__pmt_2[i] >= threshold:
                    if self.__pmt_3[i] >= threshold:
                        return True
                    else: continue
                else: continue
            else: continue
        
        return False

DataGenerator = TraceGenerator(train = True, split = 1, input_shape = 3000, fix_seed = True, shuffle = False, verbose = False)
SaveDir = "/cr/users/filip/data/first_simulation/trigger_tests/"

# have 100 signal files
for i in range(100):

    T1_actives, T2_actives, ToT_actives, actives = 0, 0, 0, 0
    traces, labels = DataGenerator.__getitem__(i)
    false_positives, false_negatives = 0, 0
    n_events = np.array([0, 0])
    trace_is_triggered = False

    print(f"Evaluating file {i + 1}/100... ", end = "")

    for trace, label in zip(traces, labels):

        EventTrigger = Trigger(trace)
        n_events += label

        if EventTrigger.T1_trigger(): T1_actives += 1; trace_is_triggered = True
        if EventTrigger.T2_trigger(): T2_actives += 1; trace_is_triggered = True
        if EventTrigger.ToT_trigger(): ToT_actives += 1; trace_is_triggered = True
        
        if trace_is_triggered: 
            actives += 1; trace_is_triggered = False
            if label[0] == 1: 
                
                false_positives += 1
                with open(SaveDir + "false_positives.csv", "a") as csv:
                    np.savetxt(csv, trace)
        else:
            if label[1] == 1: 
                
                false_negatives += 1
                with open(SaveDir + "false_negatives.csv", "a") as csv:
                    np.savetxt(csv, trace)

    with open(SaveDir + "unit_tests_summary.csv", "a") as summary_csv:
        summary_csv.write(f"signal-{str(i).zfill(3)}.csv\t\t{n_events}\t\t{T1_actives}  {T2_actives}  {ToT_actives}  {actives}\t\t{false_positives}\t\t{false_negatives}\n")

    print("DONE!")
