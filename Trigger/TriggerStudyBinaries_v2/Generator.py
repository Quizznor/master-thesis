from TriggerStudyBinaries_v2.__configure__ import *
from TriggerStudyBinaries_v2.Signal import SignalBatch, Signal
from TriggerStudyBinaries_v2.Signal import Trace, InjectedBackground
from TriggerStudyBinaries_v2.Signal import Baseline, RandomTrace

# See this website for help on a working example: shorturl.at/fFI09
class EventGenerator():

    labels = \
    {
        1: tf.keras.utils.to_categorical(1, 2, dtype = int),                    # Signal
        0: tf.keras.utils.to_categorical(0, 2, dtype = int)                     # Background
    }

    libraries = \
    {
        "19_19.5" : "/cr/tempdata01/filip/QGSJET-II/protons/19_19.5/",
        "18.5_19" : "/cr/tempdata01/filip/QGSJET-II/protons/18.5_19/",
        "18_18.5" : "/cr/tempdata01/filip/QGSJET-II/protons/18_18.5/",
        "17.5_18" : "/cr/tempdata01/filip/QGSJET-II/protons/17.5_18/",
        "17_17.5" : "/cr/tempdata01/filip/QGSJET-II/protons/17_17.5/",
        "16.5_17" : "/cr/tempdata01/filip/QGSJET-II/protons/16.5_17/",
        "16_16.5" : "/cr/tempdata01/filip/QGSJET-II/protons/16_16.5/"
    }

    def __new__(self, datasets : typing.Union[list, str], **kwargs : dict) -> typing.Union[tuple, "EventGenerator"] :

        r'''
        :datasets ``list[str]``: number of libraries you want included. Note that "all" includes everything.

        :Keyword arguments:
        
        __:Generator options:_______________________________________________________

        * *split* (``float``) -- fraction of of training set/entire set
        * *seed* (``bool``) -- fix randomizer seed for reproducibility
        * *prior* (``float``) -- p(signal), p(background) = 1 - prior

        __:VEM traces:______________________________________________________________

        * *real_background* (``bool``) -- use real background from random traces
        * *ADC_to_VEM* (``float``) -- ADC to VEM conversion factor, for UB <-> UUB
        * *n_bins* (``int``) -- generate a baseline with <trace_length> bins
        * *force_inject* (``int``) -- force the injection of <force_inject> background particles
        * *sigma* (``float``) -- baseline std in ADC counts, ignored for real_background
        * *mu* (``list``) -- mean ADC level in ADC counts, ignored for real_background

        __:Classifier:______________________________________________________________

        * *window* (``int``) -- the length of the sliding window
        * *step* (``int``) -- step size of the sliding window analysis
        * *ignore_low_VEM* (``float``) -- intentionally mislabel low signal
        '''

        # set all desired environmental variables
        split = kwargs.get("split", GLOBAL.split)
        seed = kwargs.get("seed", GLOBAL.seed)

        ADC_to_VEM = kwargs.get("ADC_to_VEM", GLOBAL.ADC_to_VEM)
        n_bins = kwargs.get("n_bins", GLOBAL.n_bins)
        baseline_std = kwargs.get("sigma", GLOBAL.baseline_std)
        baseline_mean = kwargs.get("mu", GLOBAL.baseline_mean)
        n_injected = kwargs.get("force_inject", GLOBAL.force_inject )
        real_background = kwargs.get("real_background", GLOBAL.real_background)

        ignore_low_VEM = kwargs.get("ignore_low_VEM", GLOBAL.ignore_low_VEM)
        sliding_window_length = kwargs.get("window", GLOBAL.window)
        sliding_window_step = kwargs.get("step", GLOBAL.step)

        prior = kwargs.get("prior", GLOBAL.prior)
        trace_options = [ADC_to_VEM, n_bins, baseline_std, baseline_mean, n_injected, real_background]
        classifier_options = [ignore_low_VEM, sliding_window_length, sliding_window_step, prior]
        
        # set RNG seed if desired
        if seed:
            random.seed(seed)      # does this perhaps already fix the numpy seeds?
            np.random.seed(seed)   # numpy docs says this is legacy, maybe revisit?

        # get all signal files
        if isinstance(datasets, str):
            try: data = EventGenerator.libraries[datasets]
            except KeyError:
                if datasets == "all": data = [*EventGenerator.libraries.values()]
                else: sys.exit("Couldn't construct a valid dataset from inputs")
        elif isinstance(datasets, list):
            try: data = [EventGenerator.libraries[key] for key in datasets]
            except KeyError: sys.exit("Couldn't construct a valid dataset from inputs")


        all_files = [[os.path.abspath(os.path.join(library, p)) for p in os.listdir(library)] for library in data]
        all_files = [item for sublist in all_files for item in sublist if not item.endswith("root_files")]        
        
        random.shuffle(all_files)

        # slit files into training and testing set (if needed)
        if split in [0,1]:
            return Generator(all_files, trace_options, classifier_options)
        else:
            split_files_at_index = int(split * len(all_files))
            training_files = all_files[0:split_files_at_index]
            validation_files = all_files[split_files_at_index:-1]

            TrainingSet = Generator(training_files, trace_options, classifier_options)
            TestingSet = Generator(validation_files, trace_options, classifier_options)

            return TrainingSet, TestingSet 

class Generator(tf.keras.utils.Sequence):

    def __init__(self, signal_files : list, trace_options : list, classifier_options : list) :

        self.ignore_low_VEM, self._window_length, self.window_step, self.prior = classifier_options
        self.use_real_background, trace_options = trace_options[-1], trace_options[:-1]
        self.trace_options = trace_options
        self.files = signal_files

        if self.use_real_background:
            self.RandomTraceBuffer = RandomTrace()

    def __len__(self) -> int : 
        return len(self.files)

    def __getitem__(self, index : int) -> tuple[np.ndarray] :

        labels, traces = np.array([]), np.array([])
        ADC_to_VEM, n_bins, mu, sigma, n_injected = self.trace_options

        try:
            if self.prior == 0: raise EmptyFileError
            else: event_file = self.files[index]

            for station in SignalBatch(event_file):
                
                # Build either gaussian or random trace baseline
                if not self.use_real_background: baseline = Baseline(mu, sigma, n_bins)
                else: baseline = self.RandomTraceBuffer.get()

                VEMTrace = Trace(self.trace_options, baseline, station)
                VEMTrace.__plot__()
        
        except EmptyFileError:
            print("alarm")

        print(event_file)

        return traces, labels
