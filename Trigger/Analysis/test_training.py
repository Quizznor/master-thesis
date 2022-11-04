from Binaries import *

# Ensemble1 = Ensemble("minimal_conv2d_real_background")
Ensemble2 = Ensemble("minimal_conv2d_real_background_1.00VEM")
# Ensemble3 = Ensemble("minimal_conv2d_real_background_injections")
# Ensemble4 = Ensemble("minimal_conv2d_real_background_low_prior")

Hardware = HardwareClassifier()

Hardware.ROC("random_traces")
Hardware.ROC("random_traces_downsampled")
print()
# Ensemble1.ROC("validation_data")
# print()
# Ensemble1.ROC("random_traces_downsampled")
# print()
Ensemble2.ROC("validation_data_no_cut")
print()
Ensemble2.ROC("random_traces_downsampled")
# print()
# Ensemble3.ROC("validation_data_no_injections")
# print()
# Ensemble3.ROC("random_traces_downsampled")
# print()
# Ensemble4.ROC("validation_data")
# print()
# Ensemble4.ROC("random_traces_downsampled")

plt.show()