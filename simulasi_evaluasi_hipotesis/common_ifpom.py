# common_ifpom.py

import numpy as np
import random
from copy import deepcopy

# === Fungsi Evaluasi Individu ===
def evaluate_individual(ind, projects, delta_matrix):
    Z1, Z2, Z3 = 0.0, 0.0, 0.0
    n = len(projects)

    for i in range(n):
        if ind['x'][i] == 1:
            p = projects[i]

            # Z1 - strategic value: dikalikan dengan penalti risiko
            Z1 += p['svs'] * (1 - p['risk'])

            # Z2 - risk-adjusted cost, dikoreksi oleh sinergi
            fuzzy_cost = p['fuzzy_cost']
            raw_cost = (fuzzy_cost[0] + 2 * fuzzy_cost[1] + fuzzy_cost[2]) / 4

            synergy = p['synergy_same'] + p['synergy_cross']  # sinergi sebagai pengurang biaya
            adjusted_cost = max(1.0, raw_cost - synergy)

            # kombinasi pendanaan sebagai faktor risiko biaya
            weight = (
                p['alpha'] * 0.01 +
                p['beta'] * 0.03 +
                p['theta'] * 0.08 +
                p['gamma'] * 0.01 +
                p['delta'] * 0.06
            )
            Z2 += adjusted_cost * weight * p['risk']

            # Z3 - total sinergi
            Z3 += synergy

    # Pairwise synergy (δ_ij) ditambahkan ke Z1
    lambda_val = 1.0
    for i in range(n):
        for j in range(i + 1, n):
            if ind['x'][i] == 1 and ind['x'][j] == 1:
                Z1 += lambda_val * delta_matrix[i][j]

    return [Z1, Z2, Z3]

# === Fungsi Scalarisasi: Tchebycheff ===
def tchebycheff_eval(Z, weight, ideal):
    return max([weight[i] * abs(Z[i] - ideal[i]) for i in range(len(Z))])

# === Fungsi Update Ideal Point ===
def update_ideal_point(Z, ideal):
    ideal[0] = max(ideal[0], Z[0])  # Z1 - max
    ideal[1] = min(ideal[1], Z[1])  # Z2 - min
    ideal[2] = max(ideal[2], Z[2])  # Z3 - max

# === Inisialisasi Populasi dan Parameter MOEA/D ===
def initialize_ifpom(pop_size, num_projects):
    population = []
    for _ in range(pop_size):
        # Inisialisasi x: binary project selection
        x = [random.randint(0, 1) for _ in range(num_projects)]
        if sum(x) == 0:
            x[random.randint(0, num_projects - 1)] = 1

        # Inisialisasi proporsi pendanaan (dirichlet dengan batasan theta ≤ 0.4)
        funding = [np.random.dirichlet([1, 1, 1, 1, 1]) for _ in range(num_projects)]
        ind = {
            'x': x,
            'alpha': [f[0] for f in funding],
            'beta': [f[1] for f in funding],
            'theta': [min(f[2], 0.4) for f in funding],
            'gamma': [f[3] for f in funding],
            'delta': [f[4] for f in funding],
            'Z': [None, None, None]
        }
        population.append(ind)

    # Generate weight vectors dan neighborhood untuk MOEA/D
    weight_vectors = np.random.dirichlet(np.ones(3) * 0.3, size=pop_size)
    distances = np.linalg.norm(weight_vectors[:, None, :] - weight_vectors[None, :, :], axis=2)
    neighborhoods = np.argsort(distances, axis=1)[:, :10]

    return population, weight_vectors, neighborhoods

# === Evolusi Generasi MOEA/D ===
def moead_generation(population, projects, delta_matrix, weight_vectors, neighborhoods, ideal_point):
    for i in range(len(population)):
        neighbors = neighborhoods[i]
        p1, p2 = random.sample(list(neighbors), 2)
        child = crossover(population[p1], population[p2])

        # Mutasi x
        for m in range(len(projects)):
            if random.random() < 0.1:
                child['x'][m] = 1 - child['x'][m]

        # Validasi: pastikan setidaknya satu proyek dipilih
        if sum(child['x']) == 0:
            child['x'][random.randint(0, len(projects) - 1)] = 1

        child['Z'] = evaluate_individual(child, projects, delta_matrix)

        # Replacing neighbors jika child lebih baik
        for j in neighbors:
            old_fit = tchebycheff_eval(population[j]['Z'], weight_vectors[j], ideal_point)
            new_fit = tchebycheff_eval(child['Z'], weight_vectors[j], ideal_point)
            if new_fit < old_fit:
                population[j] = deepcopy(child)

        update_ideal_point(child['Z'], ideal_point)

# === Crossover Sederhana (uniform) ===
def crossover(parent1, parent2):
    child = {'x': [], 'alpha': [], 'beta': [], 'theta': [], 'gamma': [], 'delta': [], 'Z': [None, None, None]}
    for key in ['x', 'alpha', 'beta', 'theta', 'gamma', 'delta']:
        for i in range(len(parent1[key])):
            val = parent1[key][i] if random.random() < 0.5 else parent2[key][i]
            child[key].append(val)
    return child
