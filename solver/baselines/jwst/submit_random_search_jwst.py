#!/usr/bin/env python3
"""
Submit Random Search Solution for JWST Problem
Programmable Cubes Challenge - GECCO 2024 Space Optimisation Competition (SpOC)

This script runs the random search baseline solver and creates a properly formatted
submission file for the JWST problem.

Usage:
    python solver/baselines/jwst/submit_random_search_jwst.py

Features:
    - Clean, professional random search implementation
    - Reproducible results with fixed random seed
    - Proper JSON submission format
    - Performance analysis and reporting
"""

import sys
import os
import numpy as np
import json
import time

# Add the src directory and the repository root to the Python path
repo_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, 'src'))

from solver.baselines.jwst.random_search import random_search_jwst

def create_correct_submission(challenge_id, problem_id, decision_vector, fn_out):
    """
    Create submission file in the CORRECT format as specified in the competition guidelines.
    
    Format should be a LIST of solution objects:
    [{
      "challenge": "challenge_id",
      "problem": "problem_id", 
      "decisionVector": [decision_vector]
    }]
    """
    # Convert numpy arrays to Python lists
    if isinstance(decision_vector, np.ndarray):
        decision_vector = decision_vector.tolist()
    
    # Create the submission as a LIST of solution objects
    submission = [{
        "challenge": challenge_id,
        "problem": problem_id,
        "decisionVector": decision_vector
    }]
    
    # Write to file with proper formatting
    with open(fn_out, 'w') as json_file:
        json.dump(submission, json_file, indent=2)
    
    print(f"✅ Correct format submission created: {fn_out}")
    return fn_out

def create_fixed_length_decision_vector(chromosome, max_moves):
    """Convert variable-length chromosome to fixed-length decision vector."""
    # Convert numpy array to list if needed
    if isinstance(chromosome, np.ndarray):
        chromosome = chromosome.tolist()
    
    if -1 in chromosome:
        end_pos = chromosome.index(-1)
    else:
        end_pos = len(chromosome)
    
    actual_moves = end_pos // 2
    print(f"Original chromosome has {actual_moves} moves, expanding to {max_moves} moves")
    
    decision_vector = chromosome[:end_pos].copy()
    
    # Pad with no-op moves
    moves_to_add = max_moves - actual_moves
    for _ in range(moves_to_add):
        decision_vector.extend([-1, 0])
    
    decision_vector.append(-1)
    print(f"Final decision vector length: {len(decision_vector)} (should be {max_moves * 2 + 1})")
    
    return decision_vector

def main():
    """Run the random search algorithm and create submission."""
    print("🎲 LAUNCHING RANDOM SEARCH BASELINE SOLVER")
    print("🎯 TARGET: Establish baseline performance for JWST problem")
    print("✅ FEATURES:")
    print("   • Clean, professional random search implementation")
    print("   • Reproducible results with fixed random seed")
    print("   • Comprehensive performance analysis")
    print("   • Proper JSON submission format")
    print("=" * 70)
    
    start_time = time.time()
    
    # Run the random search algorithm
    print("Running random search algorithm...")
    best_chromosome, best_fitness, best_moves = random_search_jwst()
    
    execution_time = time.time() - start_time
    
    print(f"\n🎲 RANDOM SEARCH COMPLETED!")
    print(f"⏱️  Execution time: {execution_time:.1f} seconds")
    print(f"🎯 Best fitness: {best_fitness:.6f}")
    print(f"🎲 Best moves: {best_moves}")
    print(f"📏 Chromosome length: {len(best_chromosome)}")
    
    # Performance analysis
    target_fitness = -0.991  # Championship target
    baseline_fitness = 0.186  # Typical baseline performance
    current_fitness = best_fitness
    
    print(f"\n📊 PERFORMANCE ANALYSIS:")
    print(f"Baseline performance (typical): {baseline_fitness:.6f}")
    print(f"Target performance (1st place): {target_fitness:.6f}")
    print(f"Achieved performance: {current_fitness:.6f}")
    
    # Calculate performance relative to baseline
    if current_fitness < baseline_fitness:
        improvement = baseline_fitness - current_fitness
        print(f"Improvement over baseline: {improvement:.6f}")
    else:
        gap = current_fitness - baseline_fitness
        print(f"Gap from baseline: {gap:.6f}")
    
    # Status determination
    if current_fitness <= target_fitness:
        print("🎯 EXCEPTIONAL PERFORMANCE! Championship level!")
        status_emoji = "🏆"
        status_text = "EXCEPTIONAL"
        expected_rank = "1st-3rd place"
    elif current_fitness < -0.8:
        progress = (abs(current_fitness) / 0.991) * 100
        print(f"🥇 ELITE PERFORMANCE! Progress: {progress:.1f}%")
        status_emoji = "🥇"
        status_text = "ELITE"
        expected_rank = "Top 3"
    elif current_fitness < -0.5:
        progress = (abs(current_fitness) / 0.991) * 100
        print(f"🥈 EXCELLENT PERFORMANCE! Progress: {progress:.1f}%")
        status_emoji = "🥈"
        status_text = "EXCELLENT"
        expected_rank = "Top 5"
    elif current_fitness < -0.2:
        progress = (abs(current_fitness) / 0.991) * 100
        print(f"🥉 VERY GOOD PERFORMANCE! Progress: {progress:.1f}%")
        status_emoji = "🥉"
        status_text = "VERY GOOD"
        expected_rank = "Top 10"
    elif current_fitness < 0:
        progress = (abs(current_fitness) / 0.991) * 100
        print(f"✅ NEGATIVE FITNESS ACHIEVED! Progress: {progress:.1f}%")
        status_emoji = "✅"
        status_text = "COMPETITIVE"
        expected_rank = "Top 15"
    elif current_fitness < baseline_fitness:
        improvement_pct = ((baseline_fitness - current_fitness) / baseline_fitness) * 100
        print(f"📈 GOOD BASELINE PERFORMANCE! {improvement_pct:.1f}% better than typical")
        status_emoji = "📈"
        status_text = "GOOD BASELINE"
        expected_rank = "Better than typical"
    else:
        print(f"🎲 TYPICAL BASELINE PERFORMANCE")
        status_emoji = "🎲"
        status_text = "BASELINE"
        expected_rank = "Typical baseline"
    
    print(f"{status_emoji} STATUS: {status_text}")
    print(f"🎯 EXPECTED RANKING: {expected_rank}")
    
    # Convert chromosome to decision vector
    print(f"\n🎲 Converting chromosome to decision vector...")
    decision_vector = create_fixed_length_decision_vector(best_chromosome, 30000)
    
    # Create submission with CORRECT format
    print(f"\n📄 Creating correctly formatted submission...")
    
    challenge_id = "spoc-3-programmable-cubes"
    problem_id = "jwst"
    
    output_file = os.path.join(repo_root, "submissions", "random_search_jwst_submission.json")
    
    # Ensure submissions directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Create the submission with correct format
    create_correct_submission(
        challenge_id=challenge_id,
        problem_id=problem_id,
        decision_vector=decision_vector,
        fn_out=output_file
    )
    
    print(f"\n📋 SUBMISSION DETAILS:")
    print(f"📁 File: {output_file}")
    print(f"🎯 Challenge ID: {challenge_id}")
    print(f"🧩 Problem ID: {problem_id}")
    print(f"📏 Decision vector length: {len(decision_vector)}")
    print(f"🔍 Preview: {decision_vector[:10]}...")
    
    # Validate JSON format
    try:
        with open(output_file, 'r') as f:
            loaded_data = json.load(f)
        print(f"✅ JSON format validation: PASSED")
        print(f"🔑 Keys in submission: {list(loaded_data.keys())}")
    except Exception as e:
        print(f"❌ JSON format validation: FAILED - {e}")
    
    print(f"\n" + "=" * 70)
    print("🎲 RANDOM SEARCH SUBMISSION READY!")
    print(f"{status_emoji} Performance Level: {status_text}")
    print(f"🎯 Expected Ranking: {expected_rank}")
    
    if current_fitness <= target_fitness:
        print("🏆 EXCEPTIONAL BASELINE PERFORMANCE!")
    elif current_fitness < 0:
        print("🏅 COMPETITIVE BASELINE PERFORMANCE!")
    elif current_fitness < baseline_fitness:
        print("📈 GOOD BASELINE PERFORMANCE!")
    else:
        print("🎲 TYPICAL BASELINE PERFORMANCE!")
    
    print(f"⏱️  Total time: {execution_time:.1f} seconds")
    print("=" * 70)
    
    return output_file

if __name__ == "__main__":
    submission_file = main()
