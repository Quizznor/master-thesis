from Binaries import *

TestItem = MoneyPlot()

# TestItem()

TestEnsemble = Ensemble("120_TwoLayer_FullBandwidth_AllEnergies_NoCuts")

TestItem.add(TestEnsemble, "validation_data", color = "pink")

# TestItem()

# TestEnsemble2 = Ensemble("120_TwoLayer_FullBandwidth_HighEnergies_NoCuts")

# TestItem.add(TestEnsemble2, "all_energies", color = "orange")

# # TestItem()

# TestItem.draw_line(color = "steelblue", ls = "--", lw = 4)

TestItem()