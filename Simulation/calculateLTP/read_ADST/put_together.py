#!/usr/bin/python3

import numpy as np
import sys, os

root_path = "/cr/tempdata01/filip/QGSJET-II/LTP/ADST"

energy_bins = [16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5]              # uniform in log(E)
theta_bins =  [0., 33.56, 44.42, 51.32, 56.25, 65.37]               # pseudo-uniform in sec(θ)

miss_sorted = [[ np.zeros(65) for t in range(len(theta_bins) - 1) ] for e in range(len(energy_bins) - 1)]
hits_sorted = [[ np.zeros(65) for t in range(len(theta_bins) - 1) ] for e in range(len(energy_bins) - 1)]
th1_sorted = [[ np.zeros(65) for t in range(len(theta_bins) - 1) ] for e in range(len(energy_bins) - 1)]
th2_sorted = [[ np.zeros(65) for t in range(len(theta_bins) - 1) ] for e in range(len(energy_bins) - 1)]
tot_sorted = [[ np.zeros(65) for t in range(len(theta_bins) - 1) ] for e in range(len(energy_bins) - 1)]
totd_sorted = [[ np.zeros(65) for t in range(len(theta_bins) - 1) ] for e in range(len(energy_bins) - 1)]
mops_sorted = [[ np.zeros(65) for t in range(len(theta_bins) - 1) ] for e in range(len(energy_bins) - 1)]
files_missed = 0

steps = len(os.listdir(root_path))

for k, file in enumerate(os.listdir(root_path)):

    print(f"Adding file {k}/{steps}: {100 * (k + 1)/steps:.2f}%", end = "\r")

    data = np.loadtxt(root_path + "/" + file, usecols = [1, 2, 3, 4, 5])

    #   saveFile << (i + 1) * 100 << " " << all_hits[i] << " " << misses[i] << " " << th1_hits[i] << " " << th2_hits[i] << " " << tot_hits[i] << " " << mops_hits[i] << "\n";
    _, log_e, theta, _, _, _, _ = data[0]
    hits, misses, th1, th2, tot, totd = data[1:, 0], data[1:, 1], data[1:, 2], data[1:, 3], data[1:, 4]

    t, e = np.digitize(theta, theta_bins), np.digitize(log_e, energy_bins)

    try:
        miss_sorted[e - 1][t - 1] += misses
        hits_sorted[e - 1][t - 1] += hits
        th1_sorted[e - 1][t - 1] += th1
        th2_sorted[e - 1][t - 1] += th2
        tot_sorted[e - 1][t - 1] += tot
        totd_sorted[e - 1][t - 1] += totd
    except IndexError: files_missed += 1

print("files missed:", files_missed)

# save all gathered data
for e_bin in range(1, len(energy_bins)):

    for t_bin in range(1, len(theta_bins)):

        sp_distances = np.arange(100, 6501, 100)
        save_file = f"/cr/tempdata01/filip/QGSJET-II/LTP/BINNED/{energy_bins[e_bin - 1]}_{energy_bins[e_bin]}__{int(theta_bins[t_bin - 1])}_{int(theta_bins[t_bin])}.csv"
        save_hits, save_misses = hits_sorted[e_bin - 1][t_bin - 1], miss_sorted[e_bin - 1][t_bin - 1]
        save_th1, save_th2 = th1_sorted[e_bin - 1][t_bin - 1], th2_sorted[e_bin - 1][t_bin - 1]
        save_tot, save_totd = tot_sorted[e_bin - 1][t_bin - 1], totd_sorted[e_bin - 1][t_bin - 1]
        save_mops = mops_sorted[e_bin - 1][t_bin - 1]
        save_matrix = np.dstack([sp_distances, save_hits, save_misses, save_th1, save_th2, save_tot, save_totd, save_mops])[0]

        if np.all(save_hits == 0) or np.all(save_misses == 0): continue
        np.savetxt(save_file, save_matrix)