from TriggerStudyBinaries_v2.__configure__ import *

sm = plt.cm.ScalarMappable(cmap="plasma")

# pt.spd_energy("current_trigger_validation_data", ls = "solid", label = "Th, ToT - model background", marker = "o")
pt.spd_energy("current_trigger_real_background", ls = "--", label = "Th, ToT", marker = "o")
# pt.spd_energy("mock_linear_real_background", ls = "solid", label = "Conv2d", marker = "s")
# pt.spd_energy("mock_one_real_background_uncorrected", ls = ":", label = "Conv2d, 1 VEM cut", marker = "x")
pt.spd_energy("mock_fifth_real_background_uncorrected", ls = "-.", label = "Conv2d, 1 VEM cut", marker = "v")

cbar = plt.colorbar(sm, label = "Energy")
cbar.ax.set_yticklabels(EventGenerator.libraries.keys())
plt.legend(ncol = 2)
plt.show()