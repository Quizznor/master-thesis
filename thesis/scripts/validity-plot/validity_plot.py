import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

with open('Auger_051035232600.json', "r") as file_content:  
    data = json.load(file_content)

eyes = pd.DataFrame(data['eyes'])
fdrec = pd.DataFrame(data['fdrec'])
sdrec = pd.DataFrame(data['sdrec'])

observed_energy = np.unique(sdrec['energy'])[0]

#######

for idx, eye in eyes.merge(fdrec, how='inner').iterrows():
        L = eye.uspL
        R = eye.uspR
        Xmax = eye.xmax
        dEdXmax = eye.dEdXmax
        factor = 5 * 10**8

        atm_fields = ["atmDepthProf", "energyDepositProf", "denergyDepositProf"]
        profile = pd.DataFrame(dict(zip(atm_fields, eye[atm_fields])))

        depth_observed, observed = profile.atmDepthProf, dEdXmax*pow(1 + R*(profile.atmDepthProf-Xmax)/L, pow(R,-2)) * np.exp(-(profile.atmDepthProf-Xmax) / (R*L)) * factor
        measurements, measurements_err = profile.energyDepositProf * factor, profile.denergyDepositProf * factor

depth_simulated, GAMMAS, POSITRONS, ELECTRONS, MUp, MUm, HADRONS, CHARGED, NUCLEI, CHERENKOV = np.loadtxt("gammas.txt", unpack = True)
simulated = CHARGED

depth_deterministic, nuclei, pions, electrons, photons, muons, neutrinos, remainin_energy, ionization = np.loadtxt('./showersim/result_deterministic.txt', unpack=True)
heitler_matthews_deterministic = nuclei + pions + electrons + muons

depth_stochastic, nuclei, pions, electrons, photons, muons, neutrinos, remainin_energy, ionization = np.loadtxt('./showersim/result_random.txt', unpack=True)
heitler_matthews_stochastic = nuclei + pions + electrons + muons

plt.plot((depth_deterministic[10:] - depth_deterministic[10]) / 8, heitler_matthews_deterministic[10:] * 100, label = "Heitler-Matthews", lw = 3)
plt.plot((depth_stochastic[10:] - depth_deterministic[10]) / 8, heitler_matthews_stochastic[10:] * 100, label = "HM + stochastic", lw = 3)
plt.plot(depth_observed, observed, label ="FD reconstructed", zorder = 1, lw = 3)
plt.plot(depth_simulated, simulated * 0.3, label = "QGSJETII-04", lw = 3)
plt.errorbar(depth_observed[measurements > 10**8], measurements[measurements > 10**8], yerr = measurements_err[measurements > 10**8] / 10, linestyle='', marker='o', alpha=0.2, markersize=3, linewidth=1, c = "darkgreen")

plt.plot([0, 0], [1, heitler_matthews_deterministic[10] * 100], c = "steelblue", lw = 3)
plt.plot([0, 0], [1, heitler_matthews_stochastic[10] * 100], c = "orange", lw = 3)

plt.rcParams["figure.figsize"] = [40, 15]
plt.rcParams["font.size"] = 40

plt.xlabel("Atmospheric depth / g/cm²")
plt.ylabel("Charged particles in shower")
plt.yscale("log")
plt.legend()

plt.xlabel("Atmospheric depth / g/cm²")
plt.ylabel("Shower multiplicity")
