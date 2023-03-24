import numpy as np

def apply_downsampling(pmt : np.ndarray, random_phase : int) -> np.ndarray :
    '''Receive UUB-like ADC trace and filter/downsample to emulate UB electronics'''
    n_bins_uub      = (len(pmt) // 3) * 3               # original trace length
    n_bins_ub       = n_bins_uub // 3                   # downsampled trace length
    sampled_trace   = np.zeros(n_bins_ub)               # downsampled trace container

    # ensure downsampling works as intended
    # cuts away (at most) the last two bins
    if len(pmt) % 3 != 0: pmt = pmt[0 : -(len(pmt) % 3)]

    # see Framework/SDetector/UUBDownsampleFilter.h in Offline main branch for more information
    kFirCoefficients = [ 5, 0, 12, 22, 0, -61, -96, 0, 256, 551, 681, 551, 256, 0, -96, -61, 0, 22, 12, 0, 5 ]
    buffer_length = int(0.5 * len(kFirCoefficients))
    kFirNormalizationBitShift = 11
    kADCsaturation = 4095                               # maximum FADC value: 2^12 - 1

    temp = np.zeros(n_bins_uub + len(kFirCoefficients))

    temp[0 : buffer_length] = pmt[:: -1][-buffer_length - 1 : -1]
    temp[-buffer_length - 1: -1] = pmt[:: -1][0 : buffer_length]
    temp[buffer_length : -buffer_length - 1] = pmt

    # perform downsampling
    for j, coeff in enumerate(kFirCoefficients):
        sampled_trace += [temp[k + j] * coeff for k in range(random_phase, n_bins_uub, 3)]

    # clipping and bitshifting
    sampled_trace = [int(adc) >> kFirNormalizationBitShift for adc in sampled_trace]

    # Simulate saturation of PMTs at 4095 ADC counts ~ 19 VEM LG
    return np.clip(np.array(sampled_trace), a_min = 0, a_max = kADCsaturation)