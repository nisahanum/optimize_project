from original_projects import load_project_data
from common_ifpom import (
    initialize_ifpom,
    moead_generation,
    evaluate_individual
)
from config import POPULATION_SIZE, NUM_GENERATIONS
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import json

# === Function to Save Population to JSON ===
def save_population(population, filepath="population_final.json"):
    with open(filepath, 'w') as f:
        json.dump(population, f, indent=4)
    print(f"✅ Populasi berhasil disimpan ke: {filepath}")

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

# === Load Synergy Matrix ===
delta_matrix = pd.read_csv("synergy_matrix_cosine_normalized.csv", index_col=0).values

# === MOEA/D Parameters ===
POP_SIZE = POPULATION_SIZE
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

# === Initialize Bounds ===
z_min = [float('inf')] * 3
z_max = [float('-inf')] * 3

# === Run MOEA/D ===
history = []
for gen in range(NUM_GENERATIONS):
    population, ideal_point, z_min, z_max = moead_generation(
        population, projects, delta_matrix,
        weight_vectors, neighborhoods, ideal_point,
        gen, NUM_GENERATIONS, z_min, z_max
    )

    if gen % 10 == 0:
        best_Z = max([ind['Z'][0] for ind in population]), min([ind['Z'][1] for ind in population]), max([ind['Z'][2] for ind in population])
        history.append((gen, *best_Z))
        print(f"Generation {gen}: Best Z1 = {best_Z[0]:.2f}, Z2 = {best_Z[1]:.2f}, Z3 = {best_Z[2]:.2f}")

# === Output Final Population Objectives ===
print("\n📊 Final Population Objectives:")
for i, ind in enumerate(population):
    print(f"Individual {i}: Z1={ind['Z'][0]:.2f}, Z2={ind['Z'][1]:.4f}, Z3={ind['Z'][2]:.2f}")

Z1_vals = [ind['Z'][0] for ind in population]
Z2_vals = [ind['Z'][1] for ind in population]
Z3_vals = [ind['Z'][2] for ind in population]

print("🧪 Unique Z1 values:", len(set(Z1_vals)))
print("🧪 Unique Z2 values:", len(set(Z2_vals)))
print("🧪 Unique Z3 values:", len(set(Z3_vals)))

# === Save Final Population to JSON ===
save_population(population, "population_final.json")

# === Save Final Objective Values ===
final_population_df = pd.DataFrame({
    'Z1': Z1_vals,
    'Z2': Z2_vals,
    'Z3': Z3_vals
})
#final_population_df.to_csv("final_population_objectives.csv", index=False)
#print("✅ Saved: final_population_objectives.csv")

# === Save Convergence History ===
history_df = pd.DataFrame(history, columns=['Generation', 'Z1_best', 'Z2_best', 'Z3_best'])
#history_df.to_csv("convergence_history.csv", index=False)
#print("✅ Saved: convergence_history.csv")

# === 3D Plot of Final Pareto Front ===
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Z1_vals, Z2_vals, Z3_vals, c='blue', marker='o')
ax.set_xlabel('Z1 - Strategic Value')
ax.set_ylabel('Z2 - Financial Cost')
ax.set_zlabel('Z3 - Synergy')
plt.title('Final Pareto Front (IFPOM-MOEA/D)')
plt.grid(True)
plt.show()

# === Plot Convergence Over Generations ===
generations = [h[0] for h in history]
Z1_convergence = [h[1] for h in history]
Z2_convergence = [h[2] for h in history]
Z3_convergence = [h[3] for h in history]

plt.figure(figsize=(12, 6))
plt.plot(generations, Z1_convergence, label='Z1 - Strategic Value', marker='o')
plt.plot(generations, Z2_convergence, label='Z2 - Financial Cost', marker='o')
plt.plot(generations, Z3_convergence, label='Z3 - Synergy', marker='o')
plt.xlabel("Generation")
plt.ylabel("Objective Value")
plt.title("Convergence of Z₁, Z₂, and Z₃ Over Generations")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
