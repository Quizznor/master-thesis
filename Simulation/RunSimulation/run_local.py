import os

n_simulations = 100

for i in range(n_simulations):
    os.system(f"./run_simulation.py {i}")