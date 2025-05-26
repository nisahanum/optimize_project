
from config import INITIAL_MUTATION_RATE, MIN_MUTATION_RATE, DIVERSITY_THRESHOLD, DIVERSITY_ACCEPTANCE_PROB
from tchebycheff_utils import tchebycheff_eval

import numpy as np
import random
from copy import deepcopy

def evaluate_individual(ind, projects, delta_matrix, n_samples=10):
    import numpy as np

    Z1, Z2, Z3 = 0.0, 0.0, 0.0
    n = len(projects)

    for i in range(n):
        if ind['x'][i] == 1:
            p = projects[i]

            # Z1: Strategic value adjusted by risk
            Z1 += (p['svs'] + 0.5 * (p['synergy_same'] + p['synergy_cross'])) * (1 - p['risk'])

            # === Monte Carlo fuzzy cost sampling ===
            c1, c2, c3 = p['fuzzy_cost']
            fuzzy_samples = np.random.triangular(c1, c2, c3, n_samples)

            # Funding cost multiplier based on funding mix
            funding_penalty = (
                p['alpha'] * 0.9 +    # Equity
                p['beta'] * 1.0 +     # Soft loan
                p['theta'] * 1.3 +    # Vendor more expensive
                p['gamma'] * 1.1 +    # Grant overhead
                p['delta'] * 1.2      # PPP complexity
            )

            synergy = p['synergy_same'] + p['synergy_cross']

            total_effective_cost = 0.0
            for sample_cost in fuzzy_samples:
                base_cost = max(1.0, sample_cost - synergy)
                effective_cost = base_cost * funding_penalty

                # Optional: cap runaway cost
                effective_cost = min(effective_cost, base_cost * 1.5)

                total_effective_cost += effective_cost

            avg_cost = total_effective_cost / n_samples

            # Z2: risk-adjusted average effective cost
            Z2 += avg_cost * p['risk']

            # Z3: aggregate synergy
            Z3 += synergy

    # Z1: inter-project synergy bonus
    lambda_val = 2.5
    for i in range(n):
        for j in range(i + 1, n):
            if ind['x'][i] == 1 and ind['x'][j] == 1:
                Z1 += lambda_val * delta_matrix[i][j]

    return [Z1, Z2, Z3]

def update_ideal_point(Z, ideal):
    ideal[0] = max(ideal[0], Z[0])
    ideal[1] = min(ideal[1], Z[1])
    ideal[2] = max(ideal[2], Z[2])

def initialize_ifpom(pop_size, num_projects):
    population = []
    for _ in range(pop_size):
        x = [1 if random.random() < 0.5 else 0 for _ in range(num_projects)]
        if sum(x) == 0:
            x[random.randint(0, num_projects - 1)] = 1
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
    weight_vectors = np.random.dirichlet(np.ones(3), size=pop_size)
    distances = np.linalg.norm(weight_vectors[:, None, :] - weight_vectors[None, :, :], axis=2)
    neighborhoods = np.argsort(distances, axis=1)[:, :20]
    return population, weight_vectors, neighborhoods

def crossover(parent1, parent2):
    child = {'x': [], 'alpha': [], 'beta': [], 'theta': [], 'gamma': [], 'delta': [], 'Z': [None, None, None]}
    for key in ['x', 'alpha', 'beta', 'theta', 'gamma', 'delta']:
        for i in range(len(parent1[key])):
            val = parent1[key][i] if random.random() < 0.5 else parent2[key][i]
            child[key].append(val)
    return child

def moead_generation(population, projects, delta_matrix, weight_vectors, neighborhoods, ideal_point, gen, max_gen, z_min, z_max):
    mutation_prob = INITIAL_MUTATION_RATE * (1 - gen / max_gen) + MIN_MUTATION_RATE

    for i in range(len(population)):
        neighbors = neighborhoods[i]
        p1, p2 = random.sample(list(neighbors), 2)
        child = crossover(population[p1], population[p2])

        for m in range(len(projects)):
            if random.random() < mutation_prob or child['x'][m] == population[p1]['x'][m]:
                child['x'][m] = 1 - child['x'][m]

        if sum(child['x']) == 0:
            child['x'][random.randint(0, len(projects) - 1)] = 1

        child['Z'] = evaluate_individual(child, projects, delta_matrix)

        for j in neighbors:
            old_fit = tchebycheff_eval(population[j]['Z'], weight_vectors[j], ideal_point, z_min, z_max)
            new_fit = tchebycheff_eval(child['Z'], weight_vectors[j], ideal_point, z_min, z_max)

            diversity_x = sum(child['x'][k] != population[j]['x'][k] for k in range(len(child['x'])))

            if new_fit < old_fit or (diversity_x >= DIVERSITY_THRESHOLD and random.random() < DIVERSITY_ACCEPTANCE_PROB):
                population[j] = deepcopy(child)

        update_ideal_point(child['Z'], ideal_point)

        for k in range(3):
            z_min[k] = min(z_min[k], child['Z'][k])
            z_max[k] = max(z_max[k], child['Z'][k])

    return population, ideal_point, z_min, z_max
