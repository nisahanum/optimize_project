# Re-import necessary libraries after code execution state reset
import numpy as np
import matplotlib.pyplot as plt
import random

# Problem Configuration
NUM_PROJECTS = 20
POP_SIZE = 20
LAMBDA_SYNERGY = 0.05
MAX_SELECTED = 10
NEIGHBOR_SIZE = 3
MAX_GEN = 50

# Generate project data
project_data = [{
    'roi': random.uniform(10, 100),
    'cost': random.uniform(1000, 10000),
    'synergy': random.uniform(50, 500),
    'risk': random.uniform(0.05, 0.4),
} for _ in range(NUM_PROJECTS)]

# Generate random symmetric synergy matrix delta_ij
delta_matrix = np.zeros((NUM_PROJECTS, NUM_PROJECTS))
for i in range(NUM_PROJECTS):
    for j in range(i + 1, NUM_PROJECTS):
        delta_matrix[i][j] = delta_matrix[j][i] = random.uniform(0, 100)

# Initialize individual solution
def init_solution():
    ind = []
    for _ in range(NUM_PROJECTS):
        x = 1 if random.random() < 0.3 else 0
        a = random.uniform(0, 1)
        b = random.uniform(0, 1 - a)
        c = 1 - a - b
        s = random.randint(1, 6)
        ind += [x, a, b, c, s]
    return ind

# Scalarize function
def scalarize(fitness, weights, ideal):
    return max([weights[i] * abs(fitness[i] - ideal[i]) for i in range(2)])

# Mutation
def mutate(ind):
    for i in range(0, len(ind), 5):
        if random.random() < 0.1:
            ind[i] = 1 - int(round(ind[i]))
        a = ind[i + 1]
        b = ind[i + 2]
        c = ind[i + 3]
        total = a + b + c
        if total == 0:
            a, b, c = 1.0, 0.0, 0.0
        else:
            a, b, c = a / total, b / total, c / total
        ind[i + 1] = a
        ind[i + 2] = b
        ind[i + 3] = c
    return ind

# Evaluation function with synergy
def evaluate(individual):
    total_roi = 0
    total_cost = 0
    x = []

    for i in range(NUM_PROJECTS):
        xi = int(round(individual[i * 5]))
        x.append(xi)
        a = individual[i * 5 + 1]
        b = individual[i * 5 + 2]
        c = individual[i * 5 + 3]

        if xi == 0:
            continue

        proj = project_data[i]
        ROI = proj['roi']
        cost = proj['cost']
        risk = proj['risk']

        total_roi += ROI * (1 - risk)
        total_cost += (a + b + c) * cost * (1 + risk)

    total_synergy = sum(delta_matrix[i][j] * x[i] * x[j]
                        for i in range(NUM_PROJECTS) for j in range(i + 1, NUM_PROJECTS))

    z1 = - (total_roi + LAMBDA_SYNERGY * total_synergy)
    z2 = total_cost

    if sum(x) > MAX_SELECTED:
        z2 += (sum(x) - MAX_SELECTED) * 10000

    return z1, z2

# Generate weights and neighbors
weights = [[i / (POP_SIZE - 1), 1 - i / (POP_SIZE - 1)] for i in range(POP_SIZE)]
neighbors = [sorted(range(POP_SIZE), key=lambda j: np.linalg.norm(np.array(weights[i]) - np.array(weights[j])))[:NEIGHBOR_SIZE] for i in range(POP_SIZE)]

# Initialize population
population = [init_solution() for _ in range(POP_SIZE)]
fitnesses = [evaluate(ind) for ind in population]
ideal_point = [min(f[i] for f in fitnesses) for i in range(2)]

# MOEA/D Evolution
for gen in range(MAX_GEN):
    for i in range(POP_SIZE):
        k, l = random.sample(neighbors[i], 2)
        parent1, parent2 = population[k], population[l]
        cut = random.randint(0, len(parent1))
        child = parent1[:cut] + parent2[cut:]
        child = mutate(child)
        fit_child = evaluate(child)
        for j in neighbors[i]:
            if scalarize(fit_child, weights[j], ideal_point) < scalarize(fitnesses[j], weights[j], ideal_point):
                population[j] = child
                fitnesses[j] = fit_child
                ideal_point = [min(ideal_point[d], fit_child[d]) for d in range(2)]

# Plot result
f1 = [-f[0] for f in fitnesses]
f2 = [f[1] for f in fitnesses]

plt.figure(figsize=(8, 6))
plt.scatter(f2, f1, c="orange")
plt.xlabel("Risk-Adjusted Cost (Z2)")
plt.ylabel("Strategic Value with Synergy (Z1')")
plt.title("Pareto Front with Synergy in Objective Z1")
plt.grid()
plt.show()
