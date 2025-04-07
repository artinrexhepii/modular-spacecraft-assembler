#!/usr/bin/env python3
"""
Random Search Baseline Solver for JWST Problem
Programmable Cubes Challenge - GECCO 2024 Space Optimisation Competition (SpOC)

This script implements a clean, professional baseline solver using random search
for the JWST (James Webb Space Telescope) problem. It serves as a benchmark for 
more sophisticated optimization algorithms.

Usage:
    python solver/baselines/jwst/random_search.py

Requirements:
    - numpy
    - random
    - tqdm
    - matplotlib (for plotting)
"""

import sys
import os
import numpy as np
import random
from tqdm import tqdm

# Add the src directory and the repository root to the Python path
repo_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, 'src'))

from programmable_cubes_UDP import programmable_cubes_UDP


# Configuration
N_ITERATIONS = 1000  # Increased iterations for better exploration
LOG_INTERVAL = 100   # Log progress every N iterations
RANDOM_SEED = 42    # For reproducible results


def generate_random_chromosome(num_cubes, max_length=1500):
    """
    Generate a random chromosome for the programmable cubes problem.
    
    Args:
        num_cubes (int): Number of cubes in the problem
        max_length (int): Maximum number of cube-move pairs in the chromosome
        
    Returns:
        np.ndarray: A random chromosome ending with -1
    """
    # Generate random length for the chromosome (at least 1 move)
    length = random.randint(1, max_length)
    
    chromosome = []
    for _ in range(length):
        # Random cube ID (0 to num_cubes-1)
        cube_id = random.randint(0, num_cubes - 1)
        # Random move command (0 to 5)
        move_command = random.randint(0, 5)
        
        chromosome.extend([cube_id, move_command])
    
    # Ensure chromosome ends with -1
    chromosome.append(-1)
    
    return np.array(chromosome)


def evaluate_chromosome(udp, chromosome):
    """
    Evaluate the fitness of a chromosome using the UDP.
    
    Args:
        udp: The programmable cubes UDP instance
        chromosome (list): The chromosome to evaluate
        
    Returns:
        float: The fitness score (negative value, lower is better)
    """
    try:
        fitness_score = udp.fitness(chromosome)
        return fitness_score[0]  # UDP returns a list with one element
    except Exception as e:
        print(f"Error evaluating chromosome: {e}")
        return float('-inf')  # Return worst possible fitness


def count_moves(chromosome):
    """
    Count the number of moves in a chromosome.
    
    Args:
        chromosome (np.ndarray): The chromosome
        
    Returns:
        int: Number of moves (cube-command pairs)
    """
    # Find the position of -1
    end_pos = np.where(chromosome == -1)[0][0]
    # Number of moves is half the length (since each move is cube_id + command)
    return end_pos // 2


def random_search_jwst():
    """
    Main random search algorithm for the JWST problem.
    
    Performs random search optimization and reports results.
    """
    print("=" * 60)
    print("Random Search Baseline Solver for JWST Problem")
    print("=" * 60)
    
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Initialize the UDP for JWST problem
    print("Initializing UDP for JWST problem...")
    udp = programmable_cubes_UDP('JWST')
    
    # Get problem parameters
    num_cubes = udp.setup['num_cubes']
    max_cmds = udp.setup['max_cmds']
    
    print(f"Problem parameters:")
    print(f"  - Number of cubes: {num_cubes}")
    print(f"  - Maximum commands: {max_cmds}")
    print(f"  - Number of iterations: {N_ITERATIONS}")
    print()
    
    # Initialize tracking variables
    best_fitness = float('-inf')
    best_chromosome = None
    best_moves = 0
    
    print("Starting random search...")
    print()
    
    # Main optimization loop
    for iteration in tqdm(range(N_ITERATIONS), desc="Random Search Progress"):
        # Generate random chromosome with appropriate max_length for JWST
        max_chromosome_length = min(3000, max_cmds // 10)  # Scale appropriately for JWST
        chromosome = generate_random_chromosome(num_cubes, max_length=max_chromosome_length)
        
        # Evaluate chromosome
        fitness = evaluate_chromosome(udp, chromosome)
        
        # Update best solution if this is better
        if fitness > best_fitness:
            best_fitness = fitness
            best_chromosome = chromosome.copy()
            best_moves = count_moves(chromosome)
            
            if (iteration + 1) % LOG_INTERVAL == 0:
                print(f"Iteration {iteration + 1:4d}: New best fitness = {best_fitness:.6f} (moves: {best_moves})")
        
        # Log progress periodically
        elif (iteration + 1) % LOG_INTERVAL == 0:
            print(f"Iteration {iteration + 1:4d}: Current best fitness = {best_fitness:.6f} (moves: {best_moves})")
    
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    # Print final results
    print(f"Best fitness achieved: {best_fitness:.6f}")
    print(f"Number of moves used: {best_moves}")
    print(f"Chromosome length: {len(best_chromosome)}")
    
    # Calculate and print some statistics
    efficiency = (1 - best_moves / max_cmds) * 100
    print(f"Move efficiency: {efficiency:.1f}% ({best_moves}/{max_cmds} moves used)")
    
    print()
    print("Best chromosome (first 20 elements):")
    print(best_chromosome[:20], "..." if len(best_chromosome) > 20 else "")
    
    # Plot the best solution
    print()
    print("Plotting best solution...")
    try:
        # First evaluate the best chromosome to set final_cube_positions
        udp.fitness(best_chromosome)
        
        # Plot the result
        print("Displaying ensemble configuration...")
        udp.plot('ensemble')
        
        print("Displaying target configuration...")
        udp.plot('target')
        
    except Exception as e:
        print(f"Error during plotting: {e}")
        print("Plotting may require a display. Results are still valid.")
    
    print()
    print("Random search completed successfully!")
    
    return best_chromosome, best_fitness, best_moves


if __name__ == "__main__":
    # Run the random search
    best_chromosome, best_fitness, best_moves = random_search_jwst()
