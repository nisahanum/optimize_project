# Re-run complete MOEA/D steps 1–5 after code state reset

import numpy as np
import pandas as pd
import random
from copy import deepcopy

# Parameters
num_projects = 20
pop_size = 100
T = 10
num_objectives = 3
penalty_weight = 10
mutation_prob = 0.1

np.random.seed(42)
SVS = np.random.uniform(60, 100, num_projects)
Risk = np.random.uniform(0.1, 0.5, num_projects)
Synergy = np.random.uniform(0, 20, (num_projects, num_projects))
np.fill_diagonal(Synergy, 0)

budget_limits = {'equity': 1500, 'loan1': 2000, 'loan2': 1000}
resource_limit = 1000
project_costs = np.random.uniform(50, 200, num_projects)
project_resources = np.random.uniform(10, 100, num_projects)
financing_max = {'loan1': 0.6, 'loan2': 0.4}
max_projects_selected = 10

def initialize_population(pop_size, num_projects):
    population = []
    for _ in range(pop_size):
        x = np.random.randint(0, 2, num_projects)
        funding_mix = np.random.dirichlet([1, 1, 1])
        individual = {'x': x, 'funding': funding_mix}
        population.append(individual)
    return population

def calculate_neighborhoods(weights, T):
    distances = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=2)
    return np.argsort(distances, axis=1)[:, :T]

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

    norm_Z1 = (Z1 - Z1_values.min()) / (Z1_values.max() - Z1_values.min())
    norm_Z2 = (Z2 - Z2_values.min()) / (Z2_values.max() - Z2_values.min())
    norm_Z3 = (Z3 - Z3_values.min()) / (Z3_values.max() - Z3_values.min())

    normalized_objs = np.array([1 - norm_Z1, norm_Z2, 1 - norm_Z3])
    scalar = np.max(weight_vector * np.abs(normalized_objs - z_star))
    penalized_scalar = scalar + penalty_weight * penalty

    return Z1, Z2, Z3, penalized_scalar, penalty

# Initialization
population = initialize_population(pop_size, num_projects)
weights = np.random.dirichlet(np.ones(num_objectives), size=pop_size)
neighborhoods = calculate_neighborhoods(weights, T)

# Initial objective value evaluation
Z1_raw, Z2_raw, Z3_raw = [], [], []
for ind in population:
    x = np.array(ind['x'])
    ASV = SVS * (1 - Risk)
    Z1_raw.append(np.sum(ASV * x))
    Z2_raw.append(np.sum(Risk * x))
    Z3_raw.append(sum(Synergy[i][j] * x[i] * x[j] for i in range(num_projects) for j in range(i+1, num_projects)))

Z1_values = np.array(Z1_raw)
Z2_values = np.array(Z2_raw)
Z3_values = np.array(Z3_raw)

z_star = np.min(np.vstack([
    1 - (Z1_values - Z1_values.min()) / (Z1_values.max() - Z1_values.min()),
    (Z2_values - Z2_values.min()) / (Z2_values.max() - Z2_values.min()),
    1 - (Z3_values - Z3_values.min()) / (Z3_values.max() - Z3_values.min())
]).T, axis=0)

# Initial scalar fitness computation
scalar_fitness = []
for i in range(pop_size):
    z1, z2, z3, scalar, _ = unified_evaluate(population[i], weights[i], z_star, Z1_values, Z2_values, Z3_values)
    Z1_values[i], Z2_values[i], Z3_values[i] = z1, z2, z3
    scalar_fitness.append(scalar)

# Evolutionary update loop
new_population = deepcopy(population)
for i in range(pop_size):
    neighbors = neighborhoods[i]
    p1_idx, p2_idx = np.random.choice(neighbors, 2, replace=False)
    parent1 = population[p1_idx]
    parent2 = population[p2_idx]

    child_x = [parent1['x'][j] if random.random() < 0.5 else parent2['x'][j] for j in range(num_projects)]
    child_x = [1 - bit if random.random() < mutation_prob else bit for bit in child_x]
    child = {'x': child_x, 'funding': deepcopy(parent1['funding'])}

    z1, z2, z3, scalar, penalty = unified_evaluate(child, weights[i], z_star, Z1_values, Z2_values, Z3_values)

    for j in neighborhoods[i]:
        if scalar < scalar_fitness[j]:
            new_population[j] = deepcopy(child)
            scalar_fitness[j] = scalar
            Z1_values[j], Z2_values[j], Z3_values[j] = z1, z2, z3

# Output results
final_df_integrated = pd.DataFrame({
    'Solution_ID': list(range(1, pop_size + 1)),
    'Project_Selection': [ind['x'] for ind in new_population],
    'Funding_Mix': [ind['funding'].tolist() for ind in new_population],
    'Z1': Z1_values,
    'Z2': Z2_values,
    'Z3': Z3_values,
    'Penalized_Fitness': scalar_fitness
})

print(final_df_integrated.head())
final_df_integrated.to_excel("moead_final_population_step_1_5.xlsx", index=False)