# Consolidated Python Code for MOEA/D Steps 1–4 (Initialization to Evolution)

import numpy as np
import pandas as pd
import random
from copy import deepcopy

# Parameters
num_projects = 20
pop_size = 100
T = 10
num_objectives = 3
crossover_prob = 0.9
mutation_prob = 0.1
num_generations = 1  # For demonstration; can be increased

# Step 1: Initialization
def generate_weight_vectors(pop_size, num_objectives):
    return np.random.dirichlet(np.ones(num_objectives), size=pop_size)

def calculate_neighborhoods(weights, T):
    distances = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=2)
    return np.argsort(distances, axis=1)[:, :T]

def initialize_population(pop_size, num_projects):
    population = []
    for _ in range(pop_size):
        x = np.random.randint(0, 2, num_projects)
        funding_mix = np.random.dirichlet([1, 1, 1])
        individual = {'x': x, 'funding': funding_mix}
        population.append(individual)
    return population

# Step 2: Objective evaluation
np.random.seed(42)
SVS = np.random.uniform(60, 100, num_projects)
Risk = np.random.uniform(0.1, 0.5, num_projects)
Synergy = np.random.uniform(0, 20, (num_projects, num_projects))
np.fill_diagonal(Synergy, 0)

def evaluate_objectives(individual):
    x = np.array(individual['x'])
    ASV = SVS * (1 - Risk)
    Z1 = np.sum(ASV * x)
    Z2 = np.sum(Risk * x)
    Z3 = sum(Synergy[i][j] * x[i] * x[j] for i in range(num_projects) for j in range(i+1, num_projects))
    return Z1, Z2, Z3

# Step 3: Scalar fitness calculation
def normalize_objectives(Z1, Z2, Z3):
    Z1_norm = (Z1 - Z1.min()) / (Z1.max() - Z1.min())
    Z2_norm = (Z2 - Z2.min()) / (Z2.max() - Z2.min())
    Z3_norm = (Z3 - Z3.min()) / (Z3.max() - Z3.min())
    return 1 - Z1_norm, Z2_norm, 1 - Z3_norm  # Convert to minimization

# Step 4: Evolutionary update
def bit_flip_mutation(x, prob=0.1):
    return [1 - xi if random.random() < prob else xi for xi in x]

def uniform_crossover(x1, x2):
    return [x1[i] if random.random() < 0.5 else x2[i] for i in range(len(x1))]

# Execution pipeline
weights = generate_weight_vectors(pop_size, num_objectives)
neighborhoods = calculate_neighborhoods(weights, T)
population = initialize_population(pop_size, num_projects)

Z1_list, Z2_list, Z3_list = [], [], []
for ind in population:
    z1, z2, z3 = evaluate_objectives(ind)
    Z1_list.append(z1)
    Z2_list.append(z2)
    Z3_list.append(z3)

Z1_values = np.array(Z1_list)
Z2_values = np.array(Z2_list)
Z3_values = np.array(Z3_list)

Z1_norm, Z2_norm, Z3_norm = normalize_objectives(Z1_values, Z2_values, Z3_values)
normalized_objs = np.vstack([Z1_norm, Z2_norm, Z3_norm]).T
z_star = np.min(normalized_objs, axis=0)

scalar_fitness = [np.max(weights[i] * np.abs(normalized_objs[i] - z_star)) for i in range(pop_size)]

# Evolutionary loop
new_population = deepcopy(population)
for gen in range(num_generations):
    for i in range(pop_size):
        neighbors = neighborhoods[i]
        p1_idx, p2_idx = np.random.choice(neighbors, 2, replace=False)
        parent1 = population[p1_idx]
        parent2 = population[p2_idx]

        if random.random() < crossover_prob:
            child_x = uniform_crossover(parent1['x'], parent2['x'])
        else:
            child_x = deepcopy(parent1['x'])

        child_x = bit_flip_mutation(child_x, mutation_prob)
        child = {'x': child_x, 'funding': deepcopy(parent1['funding'])}

        child_Z1, child_Z2, child_Z3 = evaluate_objectives(child)
        child_norm = normalize_objectives(
            np.append(Z1_values, child_Z1),
            np.append(Z2_values, child_Z2),
            np.append(Z3_values, child_Z3)
        )
        child_obj = np.array([child_norm[0][-1], child_norm[1][-1], child_norm[2][-1]])
        child_scalar = np.max(weights[i] * np.abs(child_obj - z_star))

        for j in neighborhoods[i]:
            if child_scalar < scalar_fitness[j]:
                new_population[j] = deepcopy(child)
                scalar_fitness[j] = child_scalar
                Z1_values[j], Z2_values[j], Z3_values[j] = child_Z1, child_Z2, child_Z3

# Final output
final_df = pd.DataFrame({
    'Solution_ID': list(range(1, pop_size + 1)),
    'Updated_Project_Selection': [ind['x'] for ind in new_population],
    'Updated_Funding_Mix': [ind['funding'].tolist() for ind in new_population],
    'Z1': Z1_values,
    'Z2': Z2_values,
    'Z3': Z3_values,
    'Tchebycheff_Fitness': scalar_fitness
})

print(final_df.head())
final_df.to_excel("moead_final_population_step_3_4.xlsx", index=False)
