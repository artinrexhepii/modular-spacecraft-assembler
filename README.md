# Modular Spacecraft Assembly Optimization

## Abstract

This repository contains the implementation and experimental framework for a bachelor thesis on modular spacecraft assembly optimization using programmable cubes. The project addresses the complex combinatorial optimization challenge of assembling spacecraft structures from modular components through intelligent cube movement sequences, as defined in the GECCO 2024 Space Optimisation Competition (SpOC).

## Problem Overview

The modular spacecraft assembly problem involves optimizing the movement sequence of programmable cubes to transform an initial 3D configuration into a target spacecraft structure. Each cube can perform pivoting maneuvers around adjacent cubes, subject to spatial constraints and movement limitations. The optimization objective is to minimize the total distance between the final and target configurations while respecting the maximum command limit.

### Problem Instances

Three distinct spacecraft assembly scenarios are studied:

- **ISS (International Space Station)**: 148 cubes, 3 cube types, 6,000 command limit
- **JWST (James Webb Space Telescope)**: 643 cubes, 6 cube types, 30,000 command limit  
- **Enterprise**: 1,472 cubes, 10 cube types, 100,000 command limit

## Methodology

### Optimization Approaches

This thesis implements and compares three distinct optimization paradigms:

#### 1. Baseline Methods (`solver/baselines/`)
- **Random Search**: Stochastic exploration baseline for performance comparison

#### 2. Heuristic Algorithms (`solver/heuristics/`)
- **Greedy Solver**: Deterministic local optimization using nearest-neighbor strategies

#### 3. Metaheuristic Optimizers (`solver/optimizers/`)
- **Genetic Algorithm**: Population-based evolutionary optimization
- **Enhanced Genetic Algorithm**: Multi-population GA with advanced features:
  - Adaptive migration strategies
  - Solution memory banking
  - Novelty-driven diversity preservation
  - Tabu-guided exploration

### Technical Framework

The implementation leverages the PyGMO optimization library with custom User-Defined Problems (UDP) for spacecraft assembly. The cube movement system supports 48 distinct pivoting maneuvers across three orthogonal planes (xy, yz, xz) with bidirectional rotation capabilities.

## Repository Structure

```
├── data/spoc3/cubes/          # Problem instance data
│   ├── Enterprise/            # Enterprise spacecraft configuration
│   ├── ISS/                   # ISS configuration  
│   └── JWST/                  # JWST configuration
├── notebooks/                 # Jupyter tutorial notebooks
├── problems/                  # Problem specification files
├── solver/                    # Optimization algorithms
│   ├── baselines/             # Random search implementations
│   ├── heuristics/            # Greedy algorithm implementations
│   ├── optimizers/            # Genetic algorithm implementations
│   └── results/               # Experimental results and visualizations
├── src/                       # Core framework components
│   ├── CubeMoveset.py         # Cube movement definitions
│   ├── programmable_cubes_UDP.py  # PyGMO problem interface
│   └── submission_helper.py   # Competition submission utilities
└── submissions/               # Generated solution files
```

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `numpy`: Numerical computing and array operations
- `matplotlib`: Visualization and plotting
- `numba`: Just-in-time compilation for performance
- `pygmo`: Optimization algorithms and problem interfaces

### Running Experiments

#### Tutorial Notebooks
Explore the framework interactively:
```bash
jupyter notebook notebooks/Tutorial--1--Running_and_analysing_the_scenarios.ipynb
```

#### Algorithm Execution
Run optimization algorithms for specific problem instances:

```bash
# Genetic Algorithm for ISS
python solver/optimizers/iss/submit_ga_iss.py

# Enhanced GA for Enterprise
python solver/optimizers/enterprise/submit_enhanced_ga_enterprise.py

# Greedy Heuristic for JWST
python solver/heuristics/jwst/submit_greedy_jwst.py
```

### Visualization and Analysis

Results are automatically generated with comprehensive visualizations:
- **Convergence plots**: Algorithm performance over iterations
- **3D assembly visualizations**: Initial, target, and final configurations
- **Ensemble comparisons**: Multiple solution candidates

Results are stored in `solver/results/[problem]/[method]/` with timestamped files.

## Key Contributions

1. **Comprehensive Algorithm Comparison**: Systematic evaluation of baseline, heuristic, and metaheuristic approaches across multiple problem scales

2. **Enhanced Genetic Algorithm**: Novel multi-population GA with advanced diversity preservation and adaptive mechanisms specifically designed for large-scale spacecraft assembly

3. **Scalability Analysis**: Investigation of algorithm performance across problem instances of varying complexity (148 to 1,472 cubes)

4. **Academic Framework**: Complete experimental infrastructure with reproducible results, statistical analysis, and academic-quality visualizations

## Experimental Results

The implemented algorithms demonstrate varying performance characteristics:

- **Random Search**: Provides baseline performance metrics for comparison
- **Greedy Algorithm**: Achieves fast convergence but may suffer from local optima
- **Genetic Algorithms**: Show superior exploration capabilities with the enhanced variant demonstrating improved convergence on large-scale problems

Detailed performance metrics and statistical analysis are available in the `solver/results/` directory.

## Future Work

- Investigation of hybrid optimization approaches combining multiple paradigms
- Development of problem-specific operators for improved search efficiency
- Extension to multi-objective optimization considering assembly time and structural stability
- Integration of machine learning techniques for adaptive parameter tuning

## Academic Context

This work contributes to the fields of:
- **Combinatorial Optimization**: Novel approaches to large-scale discrete optimization problems
- **Space Systems Engineering**: Automated assembly strategies for modular spacecraft
- **Evolutionary Computation**: Advanced genetic algorithm variants for complex engineering problems

## License

This project is developed for academic research purposes as part of a bachelor thesis.

## Acknowledgments

This work is based on the Programmable Cubes challenge from the GECCO 2024 Space Optimisation Competition (SpOC), providing a standardized framework for spacecraft assembly optimization research.