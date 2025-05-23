
from original_projects import load_project_data
from common_ifpom import (
    initialize_ifpom,
    moead_generation,
    evaluate_individual
)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === Load Project Data ===
projects = load_project_data()
NUM_PROJECTS = len(projects)

# === Compute Composite Risk for Each Project ===
for p in projects:
    risk_tech = ((9 - p['trl']) / 8) * p['complexity']
    risk_fin = (
        p['alpha'] * 0.0 +
        p['beta'] * 0.3 +
        p['theta'] * 1.0 +
        p['gamma'] * 0.1 +
        p['delta'] * 0.6
    )
    p['risk'] = max(0.05, 0.6 * risk_tech + 0.4 * risk_fin)

# === Create Dummy Delta Matrix for Pairwise Synergy ===
np.random.seed(42)
delta_matrix = np.random.uniform(0, 50, size=(NUM_PROJECTS, NUM_PROJECTS))
np.fill_diagonal(delta_matrix, 0)

# === MOEA/D Parameters ===
POP_SIZE = 50
NUM_GENERATIONS = 100
population, weight_vectors, neighborhoods = initialize_ifpom(POP_SIZE, NUM_PROJECTS)

# === Evaluate Initial Population and Set Ideal Point ===
ideal_point = [float('-inf'), float('inf'), float('-inf')]
for ind in population:
    ind['Z'] = evaluate_individual(ind, projects, delta_matrix)
    for i in range(3):
        if i == 0:
            ideal_point[i] = max(ideal_point[i], ind['Z'][i])
        elif i == 1:
            ideal_point[i] = min(ideal_point[i], ind['Z'][i])
        else:
            ideal_point[i] = max(ideal_point[i], ind['Z'][i])

# === Run MOEA/D ===
history = []
for gen in range(NUM_GENERATIONS):
    population, ideal_point = moead_generation(
        population, projects, delta_matrix, weight_vectors, neighborhoods, ideal_point, gen, NUM_GENERATIONS
    )
    if gen % 10 == 0:
        best_Z = max([ind['Z'][0] for ind in population]), min([ind['Z'][1] for ind in population]), max([ind['Z'][2] for ind in population])
        history.append((gen, *best_Z))
        print(f"Generation {gen}: Best Z1 = {best_Z[0]:.2f}, Z2 = {best_Z[1]:.2f}, Z3 = {best_Z[2]:.2f}")

# === Visualize Final Pareto Front ===
Z1_vals = [ind['Z'][0] for ind in population]
Z2_vals = [ind['Z'][1] for ind in population]
Z3_vals = [ind['Z'][2] for ind in population]

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Z1_vals, Z2_vals, Z3_vals, c='blue', marker='o')
ax.set_xlabel('Z1 - Strategic Value')
ax.set_ylabel('Z2 - Financial Cost')
ax.set_zlabel('Z3 - Synergy')
plt.title('Final Pareto Front (IFPOM-MOEA/D)')
plt.grid(True)
plt.show()
