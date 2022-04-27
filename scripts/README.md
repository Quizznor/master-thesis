* AdstExtractor/
    * functionalites for extracting VEM traces from Adsts
    * extract component traces from Adsts
* binaries/
    * Wrapper for EventGenerator, Classifier, VEMTrace, etc
* FirstModel/                                                   \
    * Dense model (2048, 12) -> 84%                             |
* SecondModel/                                                  | 
    * Dense model (4000, 100) -> 99.7%                           \  All trained with      !!
* ThirdModel/                                                    /  VEM Traces (not ADC)  !!
    * pooling layer + dense model (4000, 100) -> 99.8%          |
* FourthModel/                                                  |
    * pooling layer + convolutional layer (8, 16, 32) -> 99.7%  /
* FifthModel/
    * pooling layer + convolutional layer (8, 16, 32) -> 100%   -> trained with ADC traces