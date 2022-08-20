* TriggerStudyBinaries_v1
    * first attempt at sliding window analysis
    * Use categorical crossentropy as loss func
* TriggerStudyBinaries_v2
    * Streamline initialization of Traces
    * Modularize different kind of signals
    * Add Ensemble learning, incomplete still!
* TriggerStudyBinaries_v3
    * Optimize runtime via numba
    * Fix a bug with sliding window range
* TriggerStudyBinaries_4
    * Check out binary cross entropy -> doesn't seem to converge as good
    * Add batch normalization to input -> not worth it either (probably?)
    * Implement NNClassifier callbacks
* TriggerStudyBinaries_5
    * revert back to NOT using numba, doesn't seem to be worthwile
    * implement naive bayes classifier as a cross check
* TriggerStudyBinaries_v6
    * Correct hardware Th1 trigger to only raise T2
    * Recalculate baseline for random traces
    * Implement ToTd trigger, look at MoPS trigger