from TriggerStudyBinaries_v2.__configure__ import *

sm = plt.cm.ScalarMappable(cmap="plasma")

pt.spd_energy("current_trigger_validation_data", ls = "solid", label = "Th, ToT - model background", marker = "o")
# pt.spd_energy("current_trigger_validation_data_real_background", ls = "--", label = "Th, ToT - random traces", marker = "s")
# pt.spd_energy("mock_linear_validation_data", ls = "solid", label = "Conv2d", marker = "o")
# pt.spd_energy("mock_linear_validation_data_real_background", ls = "solid", label = "Conv2d", marker = "o")

cbar = plt.colorbar(sm, label = "Energy")
cbar.ax.set_yticklabels(EventGenerator.libraries.keys())
plt.legend(ncol = 2)
plt.show()