#!/usr/bin/env python3
"""
 Genetic Algorithm Solver for ISS Problem
Programmable Cubes Challenge - GECCO 2024 Space Optimisation Competition (SpOC)

CRITICAL FIXES:
1. ✅ FITNESS DIRECTION : Smaller/more negative fitness is better
2. ✅ Inverse-move cleanup in chromosome repair  
3. ✅ Adaptive mutation rate on stagnation
4. ✅ All operators now correctly optimize toward negative fitness

TARGET: Achieve fitness of -0.991 or better (correctly optimizing)
"""

import sys
import os
import numpy as np
import random
from tqdm import tqdm
from scipy.spatial.distance import cdist
from collections import defaultdict
import copy
import time

# Add the src directory and the repository root to the Python path
repo_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, 'src'))

from programmable_cubes_UDP import programmable_cubes_UDP

#  Configuration
POPULATION_SIZE = 100
GENERATIONS = 250
TOURNAMENT_SIZE = 5
ELITE_SIZE = 10
CROSSOVER_RATE = 0.85
BASE_MUTATION_RATE = 0.08
MAX_MUTATION_RATE = 0.25  # For adaptive mutation
MAX_CHROMOSOME_LENGTH = 800
MIN_CHROMOSOME_LENGTH = 80
RANDOM_SEED = None
LOG_INTERVAL = 20

# Initialization ratios
SMART_INITIALIZATION_RATIO = 0.5
GREEDY_INITIALIZATION_RATIO = 0.3
RANDOM_RATIO = 0.2

# Local search and cleanup
LOCAL_SEARCH_RATE = 0.3
LOCAL_SEARCH_ITERATIONS = 15
CLEANUP_RATE = 0.8  # Rate of applying inverse-move cleanup

# Adaptive parameters
STAGNATION_THRESHOLD = 25
ADAPTIVE_MUTATION_FACTOR = 1.5


class Individual:
    """Individual with  fitness handling."""
    
    def __init__(self, chromosome=None):
        self.chromosome = chromosome if chromosome is not None else []
        self.fitness = float('inf')  # Start with worst possible (positive infinity)
        self.moves_count = 0
        self.placement_accuracy = 0.0
        self.move_efficiency = 0.0
        self.is_evaluated = False
        self.age = 0
    
    def copy(self):
        """Create a deep copy of the individual."""
        new_individual = Individual(self.chromosome.copy() if self.chromosome else [])
        new_individual.fitness = self.fitness
        new_individual.moves_count = self.moves_count
        new_individual.placement_accuracy = self.placement_accuracy
        new_individual.move_efficiency = self.move_efficiency
        new_individual.is_evaluated = self.is_evaluated
        new_individual.age = self.age
        return new_individual
    
    def is_better_than(self, other):
        """Check if this individual is better (: smaller fitness is better)."""
        return self.fitness < other.fitness


def remove_inverse_moves(chromosome_seq):
    """
    Remove adjacent inverse moves for the same cube to improve efficiency.
    Inverse moves: (0,1), (2,3), (4,5) are pairs that cancel each other.
    """
    if len(chromosome_seq) < 4:  # Need at least 2 moves
        return chromosome_seq
    
    cleaned_seq = []
    i = 0
    
    while i < len(chromosome_seq) - 1:
        if i + 3 < len(chromosome_seq):  # Can check next move
            # Current move
            cube1, move1 = chromosome_seq[i], chromosome_seq[i + 1]
            # Next move  
            cube2, move2 = chromosome_seq[i + 2], chromosome_seq[i + 3]
            
            # Check if same cube and inverse moves
            if cube1 == cube2 and are_inverse_moves(move1, move2):
                # Skip both moves (they cancel out)
                i += 4
                continue
        
        # Keep this move
        cleaned_seq.extend([chromosome_seq[i], chromosome_seq[i + 1]])
        i += 2
    
    return cleaned_seq


def are_inverse_moves(move1, move2):
    """Check if two moves are inverses of each other."""
    inverse_pairs = [(0, 1), (2, 3), (4, 5)]
    for pair in inverse_pairs:
        if (move1, move2) == pair or (move2, move1) == pair:
            return True
    return False


def intelligent_cube_selection(udp, recent_moves, current_fitness=0.0):
    """Intelligent cube selection with  fitness interpretation."""
    num_cubes = udp.setup['num_cubes']
    cube_scores = np.zeros(num_cubes)
    
    for cube_id in range(num_cubes):
        score = 1.0
        
        # Penalty for recent moves
        recent_penalty = len(recent_moves[cube_id]) / 8
        score *= (1.0 - recent_penalty * 0.7)
        
        # : If fitness is good (negative), be more exploitative
        if current_fitness < 0:
            # We're doing well, be more focused
            score *= (0.8 + random.random() * 0.2)
        else:
            # We're doing poorly, be more exploratory
            score *= (0.3 + random.random() * 0.7)
        
        cube_scores[cube_id] = score
    
    # Weighted selection
    probabilities = cube_scores / np.sum(cube_scores)
    return np.random.choice(num_cubes, p=probabilities)


def intelligent_move_selection(cube_id, recent_moves, current_fitness=0.0):
    """Intelligent move selection with  fitness interpretation."""
    move_scores = np.zeros(6)
    
    for move_cmd in range(6):
        score = 1.0
        
        # Penalty for recent moves
        if move_cmd in recent_moves[cube_id]:
            score *= 0.2
        
        # : Adaptive strategy based on fitness direction
        if current_fitness < 0:
            # Good fitness, prefer tested moves
            if move_cmd % 2 == 1:  # Counterclockwise generally more effective
                score *= 1.2
        else:
            # Poor fitness, try different approaches
            score *= (0.8 + random.random() * 0.4)
        
        move_scores[move_cmd] = score
    
    # Weighted selection
    if np.sum(move_scores) > 0:
        probabilities = move_scores / np.sum(move_scores)
        return np.random.choice(6, p=probabilities)
    else:
        return random.randint(0, 5)


def generate_smart_chromosome(udp, max_length):
    """Generate smart chromosome with  optimization direction."""
    chromosome = []
    length = random.randint(MIN_CHROMOSOME_LENGTH, max_length)
    recent_moves = defaultdict(list)
    
    for _ in range(length):
        cube_id = intelligent_cube_selection(udp, recent_moves)
        move_command = intelligent_move_selection(cube_id, recent_moves)
        
        chromosome.extend([cube_id, move_command])
        
        # Update recent moves
        recent_moves[cube_id].append(move_command)
        if len(recent_moves[cube_id]) > 8:
            recent_moves[cube_id].pop(0)
    
    chromosome.append(-1)
    return chromosome


def generate_greedy_chromosome(udp, max_length):
    """Generate greedy chromosome."""
    chromosome = []
    max_moves = min(max_length, max_length)
    recent_moves = defaultdict(list)
    
    for _ in range(max_moves):
        if np.random.random() < 0.8:  # 80% greedy
            cube_id = intelligent_cube_selection(udp, recent_moves)
            move_command = intelligent_move_selection(cube_id, recent_moves)
        else:
            cube_id = random.randint(0, udp.setup['num_cubes'] - 1)
            move_command = random.randint(0, 5)
        
        chromosome.extend([cube_id, move_command])
        
        recent_moves[cube_id].append(move_command)
        if len(recent_moves[cube_id]) > 8:
            recent_moves[cube_id].pop(0)
    
    chromosome.append(-1)
    return chromosome


def generate_random_chromosome(num_cubes, max_length):
    """Generate random chromosome."""
    length = random.randint(MIN_CHROMOSOME_LENGTH, max_length)
    
    chromosome = []
    for _ in range(length):
        cube_id = random.randint(0, num_cubes - 1)
        move_command = random.randint(0, 5)
        chromosome.extend([cube_id, move_command])
    
    chromosome.append(-1)
    return chromosome


def evaluate_individual(individual, udp):
    """Evaluate individual with  fitness handling."""
    if individual.is_evaluated:
        return individual.fitness
    
    try:
        chromosome_array = np.array(individual.chromosome, dtype=int)
        fitness_score = udp.fitness(chromosome_array)
        individual.fitness = fitness_score[0]  # UDP returns negative values for good solutions
        individual.moves_count = count_moves(individual.chromosome)
        
        # Calculate detailed metrics if possible
        if hasattr(udp, 'final_cube_positions') and udp.final_cube_positions is not None:
            target_positions = udp.target_cube_positions
            final_positions = udp.final_cube_positions
            target_types = udp.target_cube_types
            initial_types = udp.initial_cube_types
            
            num_correct = 0
            total_cubes = len(final_positions)
            
            for cube_type in range(udp.setup['num_cube_types']):
                target_list = target_positions[target_types == cube_type].tolist()
                final_list = final_positions[initial_types == cube_type].tolist()
                overlap = [cube in final_list for cube in target_list]
                num_correct += np.sum(overlap)
            
            individual.placement_accuracy = num_correct / total_cubes
            individual.move_efficiency = 1.0 - (individual.moves_count / udp.setup['max_cmds'])
        
        individual.is_evaluated = True
    except Exception as e:
        print(f"Error evaluating individual: {e}")
        individual.fitness = float('inf')  # Worst possible fitness
        individual.moves_count = 0
        individual.placement_accuracy = 0.0
        individual.move_efficiency = 0.0
        individual.is_evaluated = True
    
    return individual.fitness


def count_moves(chromosome):
    """Count moves in chromosome."""
    if not chromosome:
        return 0
    try:
        end_pos = chromosome.index(-1)
        return end_pos // 2
    except ValueError:
        return len(chromosome) // 2


def initialize_population(udp, population_size):
    """Initialize population with diverse strategies."""
    population = []
    num_cubes = udp.setup['num_cubes']
    
    num_smart = int(population_size * SMART_INITIALIZATION_RATIO)
    num_greedy = int(population_size * GREEDY_INITIALIZATION_RATIO)
    num_random = population_size - num_smart - num_greedy
    
    print(f" initialization: {num_smart} smart, {num_greedy} greedy, {num_random} random")
    
    # Generate smart individuals
    for _ in range(num_smart):
        chromosome = generate_smart_chromosome(udp, MAX_CHROMOSOME_LENGTH)
        individual = Individual(chromosome)
        population.append(individual)
    
    # Generate greedy individuals
    for _ in range(num_greedy):
        chromosome = generate_greedy_chromosome(udp, MAX_CHROMOSOME_LENGTH)
        individual = Individual(chromosome)
        population.append(individual)
    
    # Generate random individuals
    for _ in range(num_random):
        chromosome = generate_random_chromosome(num_cubes, MAX_CHROMOSOME_LENGTH)
        individual = Individual(chromosome)
        population.append(individual)
    
    return population


def _tournament_selection(population, tournament_size=TOURNAMENT_SIZE):
    """ tournament selection - smaller fitness is better."""
    tournament = random.sample(population, min(tournament_size, len(population)))
    # : Return individual with SMALLEST fitness
    return min(tournament, key=lambda ind: ind.fitness)


def _crossover(parent1, parent2, udp):
    """ crossover - better parent has smaller fitness."""
    if random.random() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()
    
    try:
        # Get sequences
        end1 = parent1.chromosome.index(-1) if -1 in parent1.chromosome else len(parent1.chromosome)
        end2 = parent2.chromosome.index(-1) if -1 in parent2.chromosome else len(parent2.chromosome)
        
        seq1 = parent1.chromosome[:end1]
        seq2 = parent2.chromosome[:end2]
        
        if len(seq1) < 4 or len(seq2) < 4:
            return parent1.copy(), parent2.copy()
        
        # : Better parent has SMALLER fitness
        better_parent = parent1 if parent1.fitness < parent2.fitness else parent2
        worse_parent = parent2 if parent1.fitness < parent2.fitness else parent1
        
        # Bias toward better parent (75% vs 25%)
        better_seq = better_parent.chromosome[:better_parent.chromosome.index(-1)]
        worse_seq = worse_parent.chromosome[:worse_parent.chromosome.index(-1)]
        
        offspring1_seq = []
        offspring2_seq = []
        
        # Segment-based crossover with bias toward better parent
        min_len = min(len(better_seq), len(worse_seq))
        for i in range(0, min_len, 2):
            if random.random() < 0.75:  # Strong bias toward better parent
                if i + 1 < len(better_seq):
                    offspring1_seq.extend(better_seq[i:i+2])
                if i + 1 < len(worse_seq):
                    offspring2_seq.extend(worse_seq[i:i+2])
            else:
                if i + 1 < len(worse_seq):
                    offspring1_seq.extend(worse_seq[i:i+2])
                if i + 1 < len(better_seq):
                    offspring2_seq.extend(better_seq[i:i+2])
        
        # Add remaining from better parent
        if len(better_seq) > min_len:
            offspring1_seq.extend(better_seq[min_len:])
        
        # Create offspring
        offspring1 = Individual(repair_chromosome(offspring1_seq, udp))
        offspring2 = Individual(repair_chromosome(offspring2_seq, udp))
        
        return offspring1, offspring2
        
    except Exception:
        return parent1.copy(), parent2.copy()


def repair_chromosome(chromosome_seq, udp):
    """Enhanced chromosome repair with inverse-move cleanup."""
    if not chromosome_seq:
        return generate_random_chromosome(udp.setup['num_cubes'], MAX_CHROMOSOME_LENGTH)
    
    # Ensure even length
    if len(chromosome_seq) % 2 != 0:
        chromosome_seq = chromosome_seq[:-1]
    
    # Apply inverse-move cleanup
    if random.random() < CLEANUP_RATE:
        chromosome_seq = remove_inverse_moves(chromosome_seq)
    
    # Ensure reasonable length
    if len(chromosome_seq) < MIN_CHROMOSOME_LENGTH * 2:
        num_cubes = udp.setup['num_cubes']
        recent_moves = defaultdict(list)
        
        while len(chromosome_seq) < MIN_CHROMOSOME_LENGTH * 2:
            cube_id = intelligent_cube_selection(udp, recent_moves)
            move_cmd = intelligent_move_selection(cube_id, recent_moves)
            chromosome_seq.extend([cube_id, move_cmd])
            
            recent_moves[cube_id].append(move_cmd)
            if len(recent_moves[cube_id]) > 8:
                recent_moves[cube_id].pop(0)
    
    if len(chromosome_seq) > MAX_CHROMOSOME_LENGTH * 2:
        chromosome_seq = chromosome_seq[:MAX_CHROMOSOME_LENGTH * 2]
    
    # Validate values
    num_cubes = udp.setup['num_cubes']
    for i in range(0, len(chromosome_seq), 2):
        if i + 1 < len(chromosome_seq):
            if chromosome_seq[i] < 0 or chromosome_seq[i] >= num_cubes:
                chromosome_seq[i] = random.randint(0, num_cubes - 1)
            if chromosome_seq[i + 1] < 0 or chromosome_seq[i + 1] > 5:
                chromosome_seq[i + 1] = random.randint(0, 5)
    
    result = chromosome_seq.copy()
    result.append(-1)
    return result


def adaptive_mutation(individual, udp, mutation_rate):
    """Mutation with adaptive rate based on stagnation."""
    if random.random() > mutation_rate:
        return individual
    
    mutated = individual.copy()
    chromosome = mutated.chromosome.copy()
    
    try:
        end_pos = chromosome.index(-1) if -1 in chromosome else len(chromosome)
        move_seq = chromosome[:end_pos]
    except (ValueError, AttributeError):
        move_seq = generate_random_chromosome(udp.setup['num_cubes'], MAX_CHROMOSOME_LENGTH)[:-1]
    
    if len(move_seq) < MIN_CHROMOSOME_LENGTH * 2:
        move_seq = generate_random_chromosome(udp.setup['num_cubes'], MAX_CHROMOSOME_LENGTH)[:-1]
    
    # Number of mutations based on current mutation rate
    num_mutations = max(1, int(len(move_seq) * mutation_rate / 20))
    
    for _ in range(num_mutations):
        mutation_type = random.random()
        
        if mutation_type < 0.4 and len(move_seq) < MAX_CHROMOSOME_LENGTH * 2:
            # Insert mutation
            try:
                insert_pos = random.randrange(0, len(move_seq) + 1, 2)
                new_move = [random.randint(0, udp.setup['num_cubes'] - 1), random.randint(0, 5)]
                move_seq = move_seq[:insert_pos] + new_move + move_seq[insert_pos:]
            except ValueError:
                new_move = [random.randint(0, udp.setup['num_cubes'] - 1), random.randint(0, 5)]
                move_seq.extend(new_move)
        
        elif mutation_type < 0.7 and len(move_seq) > MIN_CHROMOSOME_LENGTH * 2:
            # Delete mutation
            try:
                delete_pos = random.randrange(0, len(move_seq), 2)
                if delete_pos + 1 < len(move_seq):
                    move_seq = move_seq[:delete_pos] + move_seq[delete_pos + 2:]
            except ValueError:
                pass
        
        else:
            # Modify mutation
            if len(move_seq) >= 2:
                try:
                    modify_pos = random.randrange(0, len(move_seq), 2)
                    if modify_pos + 1 < len(move_seq):
                        if random.random() < 0.5:
                            move_seq[modify_pos] = random.randint(0, udp.setup['num_cubes'] - 1)
                        else:
                            move_seq[modify_pos + 1] = random.randint(0, 5)
                except ValueError:
                    pass
    
    # Repair and update
    mutated.chromosome = repair_chromosome(move_seq, udp)
    mutated.is_evaluated = False
    return mutated


def local_search(individual, udp, iterations=LOCAL_SEARCH_ITERATIONS):
    """Local search for fine-tuning solutions."""
    if random.random() > LOCAL_SEARCH_RATE:
        return individual
    
    best = individual.copy()
    current = individual.copy()
    
    for _ in range(iterations):
        # Small modifications
        candidate = current.copy()
        
        if len(candidate.chromosome) > 3:
            try:
                end_pos = candidate.chromosome.index(-1)
                if end_pos >= 4:
                    pos = random.randrange(0, end_pos - 1, 2)
                    if random.random() < 0.5:
                        candidate.chromosome[pos] = random.randint(0, udp.setup['num_cubes'] - 1)
                    else:
                        candidate.chromosome[pos + 1] = random.randint(0, 5)
                    
                    candidate.is_evaluated = False
                    evaluate_individual(candidate, udp)
                    
                    # : Better means smaller fitness
                    if candidate.fitness < current.fitness:
                        current = candidate.copy()
                        if current.fitness < best.fitness:
                            best = current.copy()
            except (ValueError, IndexError):
                continue
    
    return best


def genetic_algorithm_iss():
    """ genetic algorithm with proper fitness direction."""
    print("=" * 60)
    print("🔧  Genetic Algorithm Solver for ISS Problem")
    print("✅ FITNESS DIRECTION FIXED: Smaller/negative fitness is better")
    print("✅ INVERSE-MOVE CLEANUP: Removes canceling moves")
    print("✅ ADAPTIVE MUTATION: Increases on stagnation")
    print("🎯 TARGET: Achieve fitness of -0.991 or better")
    print("=" * 60)
    
    start_time = time.time()
    
    # Initialize UDP
    print("Initializing UDP for ISS problem...")
    udp = programmable_cubes_UDP('ISS')
    
    num_cubes = udp.setup['num_cubes']
    max_cmds = udp.setup['max_cmds']
    
    print(f" parameters:")
    print(f"  - Number of cubes: {num_cubes}")
    print(f"  - Maximum commands: {max_cmds}")
    print(f"  - Population size: {POPULATION_SIZE}")
    print(f"  - Generations: {GENERATIONS}")
    print(f"  - Base mutation rate: {BASE_MUTATION_RATE}")
    print(f"  - Cleanup rate: {CLEANUP_RATE}")
    print()
    
    # Initialize population
    print("Initializing  population...")
    population = initialize_population(udp, POPULATION_SIZE)
    
    # Evaluate initial population
    print("Evaluating initial population...")
    for individual in tqdm(population, desc="Initial Evaluation"):
        evaluate_individual(individual, udp)
    
    # : Sort by fitness (smallest first)
    population.sort(key=lambda ind: ind.fitness)
    
    best_individual = population[0].copy()  # First element is now the best
    best_fitness_history = [best_individual.fitness]
    stagnation_count = 0
    current_mutation_rate = BASE_MUTATION_RATE
    
    print(f"Initial best fitness: {best_individual.fitness:.6f} (moves: {best_individual.moves_count})")
    if hasattr(best_individual, 'placement_accuracy'):
        print(f"Initial placement accuracy: {best_individual.placement_accuracy:.3f}")
    print()
    
    # Main evolution loop
    print("Starting  evolution...")
    for generation in tqdm(range(GENERATIONS), desc=" Evolution"):
        new_population = []
        
        # Age population
        for ind in population:
            ind.age += 1
        
        #  elitism: Keep the best (smallest fitness)
        elite = sorted(population, key=lambda ind: ind.fitness)[:ELITE_SIZE]
        new_population.extend([ind.copy() for ind in elite])
        
        # Generate offspring
        while len(new_population) < POPULATION_SIZE:
            #  selection
            parent1 = _tournament_selection(population)
            parent2 = _tournament_selection(population)
            
            #  crossover
            offspring1, offspring2 = _crossover(parent1, parent2, udp)
            
            # Adaptive mutation
            offspring1 = adaptive_mutation(offspring1, udp, current_mutation_rate)
            offspring2 = adaptive_mutation(offspring2, udp, current_mutation_rate)
            
            # Local search
            offspring1 = local_search(offspring1, udp)
            offspring2 = local_search(offspring2, udp)
            
            # Reset age
            offspring1.age = 0
            offspring2.age = 0
            
            new_population.extend([offspring1, offspring2])
        
        # Trim population
        new_population = new_population[:POPULATION_SIZE]
        
        # Evaluate new individuals
        for individual in new_population:
            if not individual.is_evaluated:
                evaluate_individual(individual, udp)
        
        # : Sort by fitness (smallest first)
        new_population.sort(key=lambda ind: ind.fitness)
        
        # : Check for improvement (smaller is better)
        if new_population[0].fitness < best_individual.fitness:
            improvement = best_individual.fitness - new_population[0].fitness
            best_individual = new_population[0].copy()
            stagnation_count = 0
            current_mutation_rate = BASE_MUTATION_RATE  # Reset mutation rate
            
            # Early stopping if we achieve target
            if best_individual.fitness <= -0.99:
                print(f"\n🎯 TARGET ACHIEVED! Fitness: {best_individual.fitness:.6f}")
                break
        else:
            stagnation_count += 1
            
            # Adaptive mutation: increase rate on stagnation
            if stagnation_count > STAGNATION_THRESHOLD:
                current_mutation_rate = min(MAX_MUTATION_RATE, 
                                           BASE_MUTATION_RATE * (ADAPTIVE_MUTATION_FACTOR ** (stagnation_count - STAGNATION_THRESHOLD)))
        
        best_fitness_history.append(best_individual.fitness)
        population = new_population
        
        # Restart if severely stagnated
        if stagnation_count > 50:
            print(f"\nSevere stagnation at generation {generation + 1}. Restarting...")
            # Keep top 3, regenerate rest
            super_elite = sorted(population, key=lambda ind: ind.fitness)[:3]
            new_population = initialize_population(udp, POPULATION_SIZE - 3)
            
            for ind in new_population:
                evaluate_individual(ind, udp)
            
            population = super_elite + new_population
            population.sort(key=lambda ind: ind.fitness)
            stagnation_count = 0
            current_mutation_rate = BASE_MUTATION_RATE
        
        # Logging
        if (generation + 1) % LOG_INTERVAL == 0:
            avg_fitness = np.mean([ind.fitness for ind in population])
            diversity = np.std([ind.fitness for ind in population])
            elapsed = time.time() - start_time
            
            print(f"Gen {generation + 1:3d}: Best = {best_individual.fitness:.6f}, "
                  f"Avg = {avg_fitness:.6f}, Div = {diversity:.6f}")
            print(f"         Moves = {best_individual.moves_count}, "
                  f"Stagnation = {stagnation_count}, MutRate = {current_mutation_rate:.3f}")
            
            if hasattr(best_individual, 'placement_accuracy'):
                print(f"         Accuracy = {best_individual.placement_accuracy:.3f}, "
                      f"Time = {elapsed:.1f}s")
            
            # Progress analysis
            if best_individual.fitness < 0:
                progress = (abs(best_individual.fitness) / 0.991) * 100
                print(f"         Progress toward -0.991: {progress:.1f}%")
            print()
    
    print()
    print("=" * 60)
    print("🔧  RESULTS")
    print("=" * 60)
    
    elapsed_time = time.time() - start_time
    
    # Final results
    print(f"Best fitness achieved: {best_individual.fitness:.6f}")
    print(f"Number of moves used: {best_individual.moves_count}")
    print(f"Chromosome length: {len(best_individual.chromosome)}")
    print(f"Total evolution time: {elapsed_time:.1f} seconds")
    
    # Detailed analysis
    if hasattr(best_individual, 'placement_accuracy'):
        print(f"Placement accuracy: {best_individual.placement_accuracy:.3f}")
        print(f"Move efficiency: {best_individual.move_efficiency:.3f}")
    
    # Target analysis
    target_fitness = -0.991
    current_fitness = best_individual.fitness
    
    if current_fitness <= target_fitness:
        print(f"🎯 TARGET ACHIEVED! Exceeded first place performance!")
        print(f"✅ Ready for championship submission!")
        status = "CHAMPION"
    elif current_fitness < -0.5:
        progress = (abs(current_fitness) / 0.991) * 100
        print(f"🏆 EXCELLENT PERFORMANCE! Progress: {progress:.1f}%")
        status = "ELITE"
    elif current_fitness < 0:
        progress = (abs(current_fitness) / 0.991) * 100
        print(f"✅ NEGATIVE FITNESS ACHIEVED! Progress: {progress:.1f}%")
        status = "COMPETITIVE"
    else:
        improvement = 0.186 - current_fitness  # Compare to original 9th place
        if improvement > 0:
            print(f"📈 IMPROVEMENT! Better by {improvement:.6f}")
            status = "IMPROVED"
        else:
            print(f"⚠️ Still needs work. Current: {current_fitness:.6f}")
            status = "NEEDS WORK"
    
    print(f"🏆 Status: {status}")
    
    # Show improvement
    improvement_total = best_fitness_history[0] - best_individual.fitness
    print(f"Total improvement: {improvement_total:.6f}")
    
    print()
    print("Best chromosome (first 30 elements):")
    print(best_individual.chromosome[:30], "..." if len(best_individual.chromosome) > 30 else "")
    
    print()
    print("🔧  genetic algorithm completed!")
    
    return best_individual.chromosome, best_individual.fitness, best_individual.moves_count


if __name__ == "__main__":
    best_chromosome, best_fitness, best_moves = genetic_algorithm_iss()
