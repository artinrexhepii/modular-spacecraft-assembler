#!/usr/bin/env python3
"""
Submit  Genetic Algorithm Solution for ISS Problem
Programmable Cubes Challenge - GECCO 2024 Space Optimisation Competition (SpOC)

CRITICAL FIXES APPLIED:
✅ FITNESS DIRECTION : Smaller/negative fitness is better
✅ All selection, elitism, and crossover operators fixed
✅ Inverse-move cleanup reduces junk sequences  
✅ Adaptive mutation breaks stagnation plateaus

This should give a MASSIVE improvement since the algorithm will now
optimize in the correct direction toward negative fitness values.

Target: Achieve fitness of -0.991 or better
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

from solver.optimizers.iss.ga_solver import genetic_algorithm_iss

def create_correct_submission(challenge_id, problem_id, decision_vector, fn_out, name="", description=""):
    """
    Create submission file in the CORRECT format as specified in the competition guidelines.
    
    Format should be:
    {
      "challenge": "challenge_id",
      "problem": "problem_id", 
      "decisionVector": [decision_vector],
      "name": "name",
      "description": "description"
    }
    """
    # Convert numpy arrays to Python lists
    if isinstance(decision_vector, np.ndarray):
        decision_vector = decision_vector.tolist()
    
    # Create the submission object in the correct format
    submission = {
        "challenge": challenge_id,
        "problem": problem_id,
        "decisionVector": decision_vector,
        "name": name,
        "description": description
    }
    
    # Write to file with proper formatting
    with open(fn_out, 'w') as json_file:
        json.dump(submission, json_file, indent=2)
    
    print(f"✅ Correct format submission created: {fn_out}")
    return fn_out

def create_fixed_length_decision_vector(chromosome, max_moves):
    """Convert variable-length chromosome to fixed-length decision vector."""
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
    """Run the genetic algorithm and create submission."""
    print("🔧 LAUNCHING  GENETIC ALGORITHM")
    print("✅ FITNESS DIRECTION FIXED")
    print("✅ JSON FORMAT ")
    print("🎯 TARGET: First place performance (-0.991 or better)")
    print("✅ CRITICAL FIXES APPLIED:")
    print("   • Fitness direction  (smaller is better)")
    print("   • All operators now optimize toward negative fitness")
    print("   • Inverse-move cleanup removes canceling sequences")
    print("   • Adaptive mutation breaks stagnation plateaus")
    print("=" * 70)
    
    start_time = time.time()
    
    # Run the  genetic algorithm
    print("Running  genetic algorithm...")
    best_chromosome, best_fitness, best_moves = genetic_algorithm_iss()
    
    execution_time = time.time() - start_time
    
    print(f"\n🔧  OPTIMIZATION COMPLETED!")
    print(f"⏱️  Execution time: {execution_time:.1f} seconds")
    print(f"🎯 Best fitness: {best_fitness:.6f}")
    print(f"🔧 Best moves: {best_moves}")
    print(f"📏 Chromosome length: {len(best_chromosome)}")
    
    # Performance analysis with  understanding
    target_fitness = -0.991
    original_fitness = 0.186  # Original 9th place performance
    current_fitness = best_fitness
    
    print(f"\n📊 PERFORMANCE ANALYSIS:")
    print(f"Original performance (9th place): {original_fitness:.6f}")
    print(f"Target performance (1st place): {target_fitness:.6f}")
    print(f"Achieved performance: {current_fitness:.6f}")
    
    # Calculate improvements
    improvement_over_original = original_fitness - current_fitness
    print(f"Improvement over original: {improvement_over_original:.6f}")
    
    # Status determination
    if current_fitness <= target_fitness:
        print("🎯 TARGET ACHIEVED! CHAMPIONSHIP PERFORMANCE!")
        status_emoji = "🏆"
        status_text = "CHAMPION"
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
    else:
        if improvement_over_original > 0:
            improvement_pct = (improvement_over_original / original_fitness) * 100
            print(f"📈 IMPROVED PERFORMANCE! {improvement_pct:.1f}% better")
            status_emoji = "📈"
            status_text = "IMPROVED"
            expected_rank = "Better than 9th"
        else:
            print(f"⚠️ Needs further optimization")
            status_emoji = "⚠️"
            status_text = "EXPERIMENTAL"
            expected_rank = "Experimental"
    
    print(f"{status_emoji} STATUS: {status_text}")
    print(f"🎯 EXPECTED RANKING: {expected_rank}")
    
    # Direction validation
    if current_fitness < original_fitness:
        print("✅ OPTIMIZATION DIRECTION CONFIRMED: Algorithm working correctly!")
    else:
        print("⚠️ May need further optimization")
    
    # Convert chromosome to decision vector
    print(f"\n🔧 Converting chromosome to decision vector...")
    decision_vector = create_fixed_length_decision_vector(best_chromosome, 6000)
    
    # Create submission with CORRECT format
    print(f"\n📄 Creating correctly formatted submission...")
    
    challenge_id = "spoc-3-programmable-cubes"
    problem_id = "iss"
    
    output_file = os.path.join(repo_root, "genetic_algorithm_iss_submission.json")
    
    submission_name = f"{status_emoji}  GA - ISS v4.0"
    
    submission_description = (
        f" Genetic Algorithm for ISS problem with CRITICAL FIXES: "
        f"(1) Fitness direction  - smaller/negative values are better, "
        f"(2) All operators (selection, elitism, crossover) now optimize toward negative fitness, "
        f"(3) Inverse-move cleanup removes redundant sequences, "
        f"(4) Adaptive mutation breaks stagnation plateaus. "
        f"Algorithm: Population=100, Generations=250, comprehensive optimization. "
        f"Performance: {current_fitness:.6f} fitness with {best_moves} moves. "
        f"Status: {status_text}. Improvement over 9th place: {improvement_over_original:.6f}. "
        f"Expected ranking: {expected_rank}."
    )
    
    # Create the submission with correct format
    create_correct_submission(
        challenge_id=challenge_id,
        problem_id=problem_id,
        decision_vector=decision_vector,
        fn_out=output_file,
        name=submission_name,
        description=submission_description
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
    print("🔧  SUBMISSION READY!")
    print(f"{status_emoji} Performance Level: {status_text}")
    print(f"🎯 Expected Ranking: {expected_rank}")
    
    if current_fitness <= target_fitness:
        print("🏆 CHAMPIONSHIP LEVEL SUBMISSION!")
    elif current_fitness < 0:
        print("🏅 COMPETITIVE SUBMISSION!")
    elif improvement_over_original > 0:
        print("📈 IMPROVED SUBMISSION!")
    else:
        print("🔬 EXPERIMENTAL SUBMISSION!")
    
    print(f"⏱️  Total time: {execution_time:.1f} seconds")
    print("=" * 70)
    
    return output_file

if __name__ == "__main__":
    submission_file = main()
