from Binaries import *

RandomTraces = EventGenerator(["16_16.5"], prior = 0, split = 1, real_background = True)
ModelTraces = EventGenerator(["16_16.5"], prior = 0, split = 1, real_background = False)

RandomTraces.files = np.zeros(int(1e4))
ModelTraces.files = np.zeros(int(1e4))

RandomTraces.unit_test()
ModelTraces.unit_test()