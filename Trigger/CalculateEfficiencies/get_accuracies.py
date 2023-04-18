#!/usr/bin/python3

from Binaries import *

percentages = [-16, -14, -13, -12, -11, -8, -6, -4, -2, -1, 0, 1, 2, 4, 8, 16]
percentages += list(np.arange(-50, -25, 5)) + list(np.arange(20, 101, 5))
percentages = np.array(percentages, dtype = int)

print(percentages[np.argsort(percentages)])

# percentage = percentages[int(sys.argv[1])]
# percentage_str = "m" if percentage < 0 else "p"
# percentage_str += str(percentage).replace('-', '')

# q_peak_scaled = np.array([GLOBAL.q_peak_compatibility for _ in range(3)]) * (1 + percentage * 1e-2)

# Trigger = HardwareClassifier()
# AllEvents = EventGenerator("all", apply_downsampling = True, q_peak = q_peak_scaled, seed = 42, split = 1)

# Trigger.make_signal_dataset(AllEvents, f"production_test_{percentage_str}")