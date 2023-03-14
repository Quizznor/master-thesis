import os, sys
import argparse
import numpy as np

parser = argparse.ArgumentParser("Initialize simulations for HTCondor")

parser.add_argument("dir", help = "directory (relative to ~/Simulation/) with bootstrap and condor.sub", type = str)
parser.add_argument("N", help = "How many simulations (of each energy) to run", type = int)
parser.add_argument("--t", choices = ["real", "all"], help = "What kind of trigger to employ", default = "all", type = str)
parser.add_argument("--e", help = "Which shower simulations are used", default = "all", type = str, nargs = "+")
parser.add_argument("--c", help = "Delete root files after simulation", default = False)
# ...

args = parser.parse_args()

# select all appropriate energies
all_energies = ["16_16.5", "16.5_17", "17_17.5", "17.5_18", "18_18.5", "18.5_19", "19_19.5"]
if args.e == "all": args.e = all_energies
elif ":" in args.e[0]:
    low, high = args.e[0].split(":")

    low = all_energies.index(low) if low else 0
    high = all_energies.index(high) if high else -1

    args.e = all_energies[low : high]
    args.e.append(all_energies[high])

# create target and source directory
source_path = "/cr/users/filip/Simulation/" + args.dir
target_path = "/cr/tempdata01/filip/QGSJET-II/" + args.dir

# if not os.path.exists(source_path):
#     os.makedirs(source_path)

# if not os.path.exists(target_path):
print("making dir", target_path)
try: os.makedirs(target_path)
except FileExistsError: pass

for energy in args.e:
    try: os.makedirs(target_path + "/" + energy)
    except FileExistsError: pass

        # for theta in ["0_44", "44_56", "56_65"]:
        #     try: os.makedirs(target_path + "/" + energy + "/" + theta)
        #     except FileExistsError: pass

# print(args.e)

# name = sys.argv.pop()
# print(name)