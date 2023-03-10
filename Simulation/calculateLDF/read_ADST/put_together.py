import numpy as np
import sys, os

root_path = "/cr/tempdata01/filip/QGSJET-II/LDF/ADST"

energy_bins = [16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5]              # uniform in log(E)
theta_bins =  [0., 33.56, 44.42, 51.32, 56.25, 65.37]               # pseudo-uniform in sec(θ)

miss_sorted = [[ np.zeros(65) for t in range(len(theta_bins) - 1) ] for e in range(len(energy_bins) - 1)]
hits_sorted = [[ np.zeros(65) for t in range(len(theta_bins) - 1) ] for e in range(len(energy_bins) - 1)]

for k, file in enumerate(os.listdir(root_path)):

    print(f"Adding file {k}", end = "\r")

    data = np.loadtxt(root_path + "/" + file, usecols = [1, 2])

    log_e, theta = data[0]
    hits, misses = data[1:, 0], data[1:, 1]

    t, e = np.digitize(theta, theta_bins), np.digitize(log_e, energy_bins)

    miss_sorted[e - 1][t - 1] += misses
    hits_sorted[e - 1][t - 1] += hits

# save all gathered data
for e_bin in range(1, len(energy_bins)):
    for t_bin in range(1, len(theta_bins)): 

        sp_distances = np.arange(100, 6501, 100)
        save_file = f"/cr/tempdata01/filip/QGSJET-II/LDF/BINNED/{energy_bins[e - 1]}_{energy_bins[e]}__{int(theta_bins[t_bin - 1])}_{int(theta_bins[t_bin])}.csv"
        save_hits, save_misses = hits_sorted[e_bin - 1][t_bin - 1], miss_sorted[e_bin - 1][t_bin - 1] 

        if np.all(save_hits == 0) or np.all(save_misses == 0): continue

        save_matrix = np.dstack([sp_distances, save_hits, save_misses])[0]
        np.savetxt(save_file, save_matrix)