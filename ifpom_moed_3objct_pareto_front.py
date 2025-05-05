# Revised MOEA/D implementation to improve diversity, convergence, and Pareto front quality
import numpy as np
import pandas as pd
import random
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -------------------------- Step 0: Configuration & Initialization --------------------------

# Problem parameters
num_projects = 20                # Total number of candidate projects
pop_size = 100                   # Population size (number of solutions)
T = 10                           # Number of neighbors in MOEA/D
num_objectives = 3               # We optimize Z1, Z2, Z3
penalty_weight = 10             # Penalty multiplier for constraint violations
mutation_prob = 0.2             # Probability of bit-flip mutation (lower than old code for stability)
num_generations = 50            # Increased number of generations (from 1 → 50)

# Set seed for reproducibility
np.random.seed(42)

# Data simulation
SVS = np.random.uniform(60, 100, num_projects)  # Strategic Value Score
Risk = np.random.uniform(0.1, 0.5, num_projects)  # Project risk
Synergy = np.random.uniform(0, 20, (num_projects, num_projects))  # Pairwise synergy
np.fill_diagonal(Synergy, 0)

# Constraints
budget_limits = {'equity': 1500, 'loan1': 2000, 'loan2': 1000}
resource_limit = 1000
project_costs = np.random.uniform(50, 200, num_projects)
project_resources = np.random.uniform(10, 100, num_projects)
financing_max = {'loan1': 0.6, 'loan2': 0.4}
max_projects_selected = 10

# -------------------------- Step 1: Helper Functions --------------------------

# Initialize population: each solution includes binary selection vector and funding mix
def initialize_population(pop_size, num_projects):
    population = []
    for _ in range(pop_size):
        x = np.random.randint(0, 2, num_projects)
        funding_mix = np.random.dirichlet([1, 1, 1])  # equity, loan1, loan2
        individual = {'x': x, 'funding': funding_mix}
        population.append(individual)
    return population

# Calculate T-nearest neighbors using Euclidean distance in weight space
def calculate_neighborhoods(weights, T):
    distances = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=2)
    return np.argsort(distances, axis=1)[:, :T]

# Unified evaluation function with penalty and scalarizing fitness
def unified_evaluate(individual, weight_vector, z_star, Z1_values, Z2_values, Z3_values):
    x = np.array(individual['x'])
    ASV = SVS * (1 - Risk)
    Z1 = np.sum(ASV * x)
    Z2 = np.sum(Risk * x)
    Z3 = sum(Synergy[i][j] * x[i] * x[j] for i in range(num_projects) for j in range(i+1, num_projects))

    cost = project_costs * x
    res_usage = np.sum(project_resources * x)
    alpha, beta, gamma = individual['funding']
    funding_valid = (abs(alpha + beta + gamma - 1) <= 0.01) and (beta <= financing_max['loan1']) and (gamma <= financing_max['loan2'])

    budget_used = {
        'equity': np.sum(alpha * cost),
        'loan1': np.sum(beta * cost),
        'loan2': np.sum(gamma * cost)
    }

    penalty = 0
    if budget_used['equity'] > budget_limits['equity']:
        penalty += (budget_used['equity'] - budget_limits['equity']) / budget_limits['equity']
    if budget_used['loan1'] > budget_limits['loan1']:
        penalty += (budget_used['loan1'] - budget_limits['loan1']) / budget_limits['loan1']
    if budget_used['loan2'] > budget_limits['loan2']:
        penalty += (budget_used['loan2'] - budget_limits['loan2']) / budget_limits['loan2']
    if res_usage > resource_limit:
        penalty += (res_usage - resource_limit) / resource_limit
    if not funding_valid:
        penalty += 1
    if np.sum(x) > max_projects_selected:
        penalty += (np.sum(x) - max_projects_selected) / max_projects_selected

    # Normalize objective values with small constant added to avoid division by zero
    norm_Z1 = (Z1 - Z1_values.min()) / (Z1_values.max() - Z1_values.min() + 1e-6)
    norm_Z2 = (Z2 - Z2_values.min()) / (Z2_values.max() - Z2_values.min() + 1e-6)
    norm_Z3 = (Z3 - Z3_values.min()) / (Z3_values.max() - Z3_values.min() + 1e-6)

    # Tchebycheff scalarization (min-max distance to ideal point)
    normalized_objs = np.array([1 - norm_Z1, norm_Z2, 1 - norm_Z3])
    scalar = np.max(weight_vector * np.abs(normalized_objs - z_star))
    penalized_scalar = scalar + penalty_weight * penalty
    return Z1, Z2, Z3, penalized_scalar

# -------------------------- Step 2: Initialization --------------------------

population = initialize_population(pop_size, num_projects)
weights = np.random.dirichlet(np.ones(num_objectives), size=pop_size)
neighborhoods = calculate_neighborhoods(weights, T)

Z1_values = np.zeros(pop_size)
Z2_values = np.zeros(pop_size)
Z3_values = np.zeros(pop_size)
scalar_fitness = np.zeros(pop_size)

# -------------------------- Step 3: Evolutionary Loop --------------------------

for generation in range(num_generations):
    for i in range(pop_size):
        ind = population[i]
        x = np.array(ind['x'])
        ASV = SVS * (1 - Risk)
        Z1_values[i] = np.sum(ASV * x)
        Z2_values[i] = np.sum(Risk * x)
        Z3_values[i] = sum(Synergy[p][q] * x[p] * x[q] for p in range(num_projects) for q in range(p+1, num_projects))

    # Ideal point update
    z_star = np.min(np.vstack([
        1 - (Z1_values - Z1_values.min()) / (Z1_values.max() - Z1_values.min() + 1e-6),
        (Z2_values - Z2_values.min()) / (Z2_values.max() - Z2_values.min() + 1e-6),
        1 - (Z3_values - Z3_values.min()) / (Z3_values.max() - Z3_values.min() + 1e-6)
    ]).T, axis=0)

    # Mating + replacement per neighborhood
    new_population = deepcopy(population)
    for i in range(pop_size):
        neighbors = neighborhoods[i]
        p1_idx, p2_idx = np.random.choice(neighbors, 2, replace=False)
        parent1 = population[p1_idx]
        parent2 = population[p2_idx]

        # Crossover and mutation
        child_x = [parent1['x'][j] if random.random() < 0.5 else parent2['x'][j] for j in range(num_projects)]
        child_x = [1 - bit if random.random() < mutation_prob else bit for bit in child_x]
        child = {'x': child_x, 'funding': deepcopy(parent1['funding'])}

        # Evaluate and decide replacement
        z1, z2, z3, scalar = unified_evaluate(child, weights[i], z_star, Z1_values, Z2_values, Z3_values)
        if scalar < scalar_fitness[i] or generation == 0:
            new_population[i] = deepcopy(child)
            Z1_values[i], Z2_values[i], Z3_values[i] = z1, z2, z3
            scalar_fitness[i] = scalar

    population = new_population

# -------------------------- Step 4: Pareto Front Extraction --------------------------

def is_dominated(point, others):
    return any(all(o <= p for o, p in zip(other, point)) and any(o < p for o, p in zip(other, point)) for other in others)

def identify_pareto_front(Z1, Z2, Z3):
    points = np.vstack((Z1, Z2, Z3)).T
    pareto_mask = np.ones(points.shape[0], dtype=bool)
    for i, point in enumerate(points):
        others = np.delete(points, i, axis=0)
        if is_dominated(point, others):
            pareto_mask[i] = False
    return pareto_mask

pareto_mask = identify_pareto_front(Z1_values, Z2_values, Z3_values)
pareto_front_df = pd.DataFrame({
    'Z1': Z1_values[pareto_mask],
    'Z2': Z2_values[pareto_mask],
    'Z3': Z3_values[pareto_mask]
})

# -------------------------- Step 5: Save to Excel --------------------------

# pareto_front_df.to_excel("moead_final_pareto_front_visualitation_ver2.xlsx", index=False)

# -------------------------- Step 6: Visualization --------------------------

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Z1_values, Z2_values, Z3_values, c='gray', label='All Solutions', alpha=0.3)
ax.scatter(pareto_front_df['Z1'], pareto_front_df['Z2'], pareto_front_df['Z3'],
           c='red', label='Pareto Front', s=60)
ax.set_xlabel('Z1: Strategic Value')
ax.set_ylabel('Z2: Risk Exposure')
ax.set_zlabel('Z3: Synergy Value')
ax.set_title('MOEA/D Improved Pareto Front Visualization')
ax.legend()
plt.tight_layout()
plt.show()
