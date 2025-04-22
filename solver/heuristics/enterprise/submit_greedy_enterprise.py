#!/usr/bin/env python3
"""
Submit Greedy Solver Solution for Enterprise Problem
Programmable Cubes Challenge - GECCO 2024 Space Optimisation Competition (SpOC)

This script runs the greedy solver for the Enterprise problem and creates a submission file.
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

from solver.heuristics.enterprise.greedy_solver import greedy_search_enterprise

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
    """
    Convert variable-length chromosome to fixed-length decision vector.
    
    Args:
        chromosome (np.ndarray): Variable-length chromosome ending with -1
        max_moves (int): Maximum number of moves (100000 for Enterprise)
        
    Returns:
        list: Fixed-length decision vector of length (max_moves * 2 + 1)
    """
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
    """
    Run the greedy solver and create a submission file for the Enterprise problem.
    """
    print("🎯 LAUNCHING GREEDY HEURISTIC SOLVER")
    print("🎯 TARGET: Establish competitive performance for Enterprise problem")
    print("✅ FEATURES:")
    print("   • Balanced greedy/random strategy (70% greedy, 30% random)")
    print("   • Recent move tracking to avoid redundancy")
    print("   • Probabilistic selection for exploration")
    print("   • Proper JSON submission format")
    print("=" * 70)
    
    start_time = time.time()
    
    # Run the greedy solver
    print("Running greedy solver for Enterprise problem...")
    best_chromosome, best_fitness, best_moves = greedy_search_enterprise()
    
    execution_time = time.time() - start_time
    
    print(f"\n🎯 GREEDY SOLVER COMPLETED!")
    print(f"⏱️  Execution time: {execution_time:.1f} seconds")
    print(f"🎯 Best fitness: {best_fitness:.6f}")
    print(f"🎲 Best moves: {best_moves}")
    print(f"📏 Chromosome length: {len(best_chromosome)}")
    
    # Performance analysis for Enterprise
    target_fitness = -0.8  # Championship target for Enterprise
    baseline_fitness = 0.015  # Random search baseline for Enterprise
    current_fitness = best_fitness
    
    print(f"\n📊 PERFORMANCE ANALYSIS:")
    print(f"Random search baseline: {baseline_fitness:.6f}")
    print(f"Target performance (1st place): {target_fitness:.6f}")
    print(f"Achieved performance: {current_fitness:.6f}")
    
    # Calculate improvement over random search
    if current_fitness < baseline_fitness:
        improvement = baseline_fitness - current_fitness
        improvement_pct = (improvement / baseline_fitness) * 100
        print(f"Improvement over random search: {improvement:.6f} ({improvement_pct:.1f}%)")
    else:
        gap = current_fitness - baseline_fitness
        print(f"Gap from random search baseline: {gap:.6f}")
    
    # Status determination for Enterprise
    if current_fitness <= target_fitness:
        print("🎯 EXCEPTIONAL PERFORMANCE! Championship level!")
        status_emoji = "🏆"
        status_text = "EXCEPTIONAL"
        expected_rank = "1st-3rd place"
    elif current_fitness < -0.5:
        progress = (abs(current_fitness) / 0.8) * 100
        print(f"🥇 ELITE PERFORMANCE! Progress: {progress:.1f}%")
        status_emoji = "🥇"
        status_text = "ELITE"
        expected_rank = "Top 3"
    elif current_fitness < -0.2:
        progress = (abs(current_fitness) / 0.8) * 100
        print(f"🥈 EXCELLENT PERFORMANCE! Progress: {progress:.1f}%")
        status_emoji = "🥈"
        status_text = "EXCELLENT"
        expected_rank = "Top 5"
    elif current_fitness < 0:
        progress = (abs(current_fitness) / 0.8) * 100
        print(f"🥉 VERY GOOD PERFORMANCE! Progress: {progress:.1f}%")
        status_emoji = "🥉"
        status_text = "VERY GOOD"
        expected_rank = "Top 10"
    elif current_fitness < 0.2:
        progress = (abs(current_fitness) / 0.8) * 100
        print(f"✅ NEGATIVE FITNESS ACHIEVED! Progress: {progress:.1f}%")
        status_emoji = "✅"
        status_text = "COMPETITIVE"
        expected_rank = "Top 15"
    elif current_fitness < baseline_fitness:
        improvement_pct = ((baseline_fitness - current_fitness) / baseline_fitness) * 100
        print(f"📈 GOOD BASELINE PERFORMANCE! {improvement_pct:.1f}% better than random search")
        status_emoji = "📈"
        status_text = "GOOD BASELINE"
        expected_rank = "Better than random search"
    else:
        print(f"🎲 TYPICAL BASELINE PERFORMANCE")
        status_emoji = "🎲"
        status_text = "BASELINE"
        expected_rank = "Typical baseline"
    
    print(f"{status_emoji} STATUS: {status_text}")
    print(f"🎯 EXPECTED RANKING: {expected_rank}")
    
    # Convert chromosome to decision vector
    print(f"\n🎯 Converting chromosome to decision vector...")
    decision_vector = create_fixed_length_decision_vector(best_chromosome, 100000)
    
    # Create submission with CORRECT format
    print(f"\n📄 Creating correctly formatted submission...")
    
    challenge_id = "spoc-3-programmable-cubes"
    problem_id = "enterprise"
    
    output_file = os.path.join(repo_root, "submissions", "greedy_enterprise_submission.json")
    
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
        print(f"🔑 Submission structure: {type(loaded_data)} with {len(loaded_data)} solution(s)")
        if len(loaded_data) > 0:
            print(f"🔑 Keys in first solution: {list(loaded_data[0].keys())}")
    except Exception as e:
        print(f"❌ JSON format validation: FAILED - {e}")
    
    print(f"\n" + "=" * 70)
    print("🎯 GREEDY SOLVER SUBMISSION READY!")
    print(f"{status_emoji} Performance Level: {status_text}")
    print(f"🎯 Expected Ranking: {expected_rank}")
    
    if current_fitness <= target_fitness:
        print("🏆 EXCEPTIONAL GREEDY PERFORMANCE!")
    elif current_fitness < 0:
        print("🏅 COMPETITIVE GREEDY PERFORMANCE!")
    elif current_fitness < baseline_fitness:
        print("📈 GOOD GREEDY PERFORMANCE!")
    else:
        print("🎲 TYPICAL GREEDY PERFORMANCE!")
    
    print(f"⏱️  Total time: {execution_time:.1f} seconds")
    print("=" * 70)
    
    return output_file

if __name__ == "__main__":
    submission_file = main()
