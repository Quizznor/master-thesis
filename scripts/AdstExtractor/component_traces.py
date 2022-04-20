import numpy as np
from pyik.adst import *

FILE="/lsdf/auger/tmp/hahn/test_production/napoli/ub/QGSJET-II.04/proton/19_19.5/01/DAT991801_00_adst.root"


# for ev in pyik.adst.RecEventProvider(FILE):
#     print(ev)


# def GetWCDADCTraces(station, component=0):
#     traces = []
#     for iPMT in range(3):
#         pmt_traces = station.GetPMTTraces(component,iPMT+1)
#         calibrated_trace = np.array(pmt_traces.GetVEMComponent())
#         vem_peak = pmt_traces.GetPeak()
#         uncalibrated_trace = calibrated_trace*vem_peak
#         if station.IsHighGainSaturated():
#             dynode_anode_ratio = station.GetDynodeAnodeRatio(iPMT+1)
#             uncalibrated_trace = uncalibrated_trace / dynode_anode_ratio
#         traces.append(uncalibrated_trace)
#     return traces