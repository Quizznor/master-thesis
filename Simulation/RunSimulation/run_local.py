import os

n_simulations = 5012

for i in range(n_simulations):
    os.system(f"./run_simulation.py {i}")