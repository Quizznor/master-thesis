#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np

# observed shower
# /lsdf/auger/corsika/napoli/QGSJET-II.04/proton/18.5_19/DAT784868.lst

# E = 8.54718e+09

# Observed
# depth_ovserved, observed = np.loadtxt()

# Simulated
depth_simulated, GAMMAS, POSITRONS, ELECTRONS, MUp, MUm, HADRONS, CHARGED, NUCLEI, CHERENKOV = np.loadtxt("gammas.txt", unpack = True)
simulated = GAMMAS + POSITRONS + ELECTRONS + MUp + MUm + HADRONS + CHARGED + NUCLEI + CHERENKOV

# Heitler-Matthews
depth_deterministic, nuclei, pions, electrons, photons, muons, neutrinos, remainin_energy, ionization = np.loadtxt('./showersim/result_deterministic.txt', unpack=True)
heitler_matthews_deterministic = nuclei + pions + electrons + photons + muons

# Heitler-Matthews + stochastic component
depth_stochastic, nuclei, pions, electrons, photons, muons, neutrinos, remainin_energy, ionization = np.loadtxt('./showersim/result_random.txt', unpack=True)
heitler_matthews_stochastic = nuclei + pions + electrons + photons + muons


plt.plot(depth_deterministic, heitler_matthews_deterministic, label = "Heitler-Matthews")
plt.plot(depth_stochastic, heitler_matthews_stochastic, label = "HM + stochastic")
plt.plot(depth_simulated * 5, simulated, label = "CORSIKA, QGSJET-II.04")
# plt.plot(depth_observed, observed, label ="Observed")

plt.yscale("log")
plt.legend()
plt.show()