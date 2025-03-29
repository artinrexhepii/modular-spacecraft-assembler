import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np

from src.programmable_cubes_UDP import programmable_cubes_UDP

# Load ISS problem to test setup
udp = programmable_cubes_UDP('ISS')

# Print number of decision variables
print(f"Number of decision variables: {udp.get_nix()}")



chromosome = np.array([0, 0, 1, 1, -1])

# Evaluate fitness
fitness = udp.fitness(chromosome)
print(f"Fitness of dummy chromosome: {fitness}")

# Plot target structure
udp.plot("target")

# Plot ensemble after dummy chromosome
udp.plot("ensemble")