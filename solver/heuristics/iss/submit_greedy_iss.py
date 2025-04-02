#!/usr/bin/env python3
"""
Submit Greedy Solver Solution for ISS Problem
Programmable Cubes Challenge - GECCO 2024 Space Optimisation Competition (SpOC)

This script runs the greedy solver for the ISS problem and creates a submission file.
"""

import sys
import os
import numpy as np

# Add the src directory and the repository root to the Python path
repo_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, 'src'))

from solver.heuristics.iss.greedy_solver import greedy_search_iss
from submission_helper import create_submission

def create_fixed_length_decision_vector(chromosome, max_moves):
    """
    Convert variable-length chromosome to fixed-length decision vector.
    
    Args:
        chromosome (np.ndarray): Variable-length chromosome ending with -1
        max_moves (int): Maximum number of moves (6000 for ISS)
        
    Returns:
        list: Fixed-length decision vector of length (max_moves * 2 + 1)
    """
    # Find the position of -1 (end of actual moves)
    end_pos = np.where(chromosome == -1)[0][0]
    actual_moves = end_pos // 2  # Number of actual moves
    
    print(f"Original chromosome has {actual_moves} moves, expanding to {max_moves} moves")
    
    # Start with the actual moves from the chromosome
    decision_vector = chromosome[:end_pos].tolist()
    
    # Pad with no-op moves to reach exactly max_moves
    # Use cube_id=0, move_id=0 as no-op (or any valid values)
    moves_to_add = max_moves - actual_moves
    for _ in range(moves_to_add):
        decision_vector.extend([0, 0])  # No-op move: cube 0, move 0
    
    # Add terminating -1
    decision_vector.append(-1)
    
    print(f"Final decision vector length: {len(decision_vector)} (should be {max_moves * 2 + 1})")
    
    return decision_vector

def main():
    """
    Run the greedy solver and create a submission file for the ISS problem.
    """
    print("Running greedy solver for ISS problem...")
    
    # Run the greedy solver
    best_chromosome, best_fitness, best_moves = greedy_search_iss()
    
    print(f"\nGreedy solver completed!")
    print(f"Best fitness: {best_fitness:.6f}")
    print(f"Best moves: {best_moves}")
    print(f"Chromosome length: {len(best_chromosome)}")
    
    # Convert chromosome to proper decision vector format
    # ISS requires exactly 12001 elements: 6000 moves (cube_id, move_id pairs) + terminating -1
    decision_vector = create_fixed_length_decision_vector(best_chromosome, 6000)
    
    # Create submission file
    # Based on the tutorial notebook, the correct IDs are:
    challenge_id = "spoc-3-programmable-cubes"  # Space Optimisation Competition
    problem_id = "iss"     # ISS problem identifier (lowercase)
    
    output_file = os.path.join(repo_root, "greedy_iss_submission.json")
    
    create_submission(
        challenge_id=challenge_id,
        problem_id=problem_id,
        x=decision_vector,
        fn_out=output_file,
        name="Greedy Heuristic Solver - ISS",
        description=f"Greedy heuristic solver for ISS problem. Achieved fitness: {best_fitness:.6f} with {best_moves} moves. Uses balanced greedy/random strategy with 70% greedy selection and recent move tracking."
    )
    
    print(f"\nSubmission file created: {output_file}")
    print(f"Challenge ID: {challenge_id}")
    print(f"Problem ID: {problem_id}")
    print(f"Decision vector length: {len(decision_vector)}")
    
    # Show first few elements of the decision vector
    print(f"Decision vector (first 20 elements): {decision_vector[:20]}")
    
    return output_file

if __name__ == "__main__":
    submission_file = main()
