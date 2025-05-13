import numpy as np
import pandas as pd

# Parameters
num_projects = 20              # number of projects in the portfolio
pop_size = 100                 # number of sub-problems (i.e., individuals)
T = 10                         # neighborhood size
num_objectives = 3            # Z1, Z2, Z3

# Step 1: Generate weight vectors (lambda) for decomposition
def generate_weight_vectors(pop_size, num_objectives):
    weights = np.random.dirichlet(np.ones(num_objectives), size=pop_size)
    return weights

# Step 2: Compute neighborhood matrix based on Euclidean distance in weight space
def calculate_neighborhoods(weights, T):
    distances = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=2)
    neighborhoods = np.argsort(distances, axis=1)[:, :T]
    return neighborhoods

# Step 3: Initialize population
def initialize_population(pop_size, num_projects):
    population = []
    for _ in range(pop_size):
        # Binary project selection vector (0/1)
        x = np.random.randint(0, 2, num_projects)
        # Real-valued funding mix per project (just one for simplicity now)
        funding_mix = np.random.dirichlet([1, 1, 1])  # α, β, γ
        individual = {'x': x, 'funding': funding_mix}
        population.append(individual)
    return population

# Execute initialization
weights = generate_weight_vectors(pop_size, num_objectives)
neighborhoods = calculate_neighborhoods(weights, T)
population = initialize_population(pop_size, num_projects)

# Optional: Create a summary DataFrame
population_df = pd.DataFrame({
    'Solution_ID': list(range(1, pop_size + 1)),
    'Project_Selection': [ind['x'].tolist() for ind in population],
    'Funding_Mix': [ind['funding'].tolist() for ind in population],
    'Neighbors': neighborhoods.tolist()
})

# Print a few sample rows nicely
for i in range(5):
    print(f"Solution {i+1}:")
    print("  Project Selection:", population[i]['x'])
    print("  Funding Mix:", population[i]['funding'])
    print("  Neighbors:", neighborhoods[i])
    print()