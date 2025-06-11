
from config import INITIAL_MUTATION_RATE, MIN_MUTATION_RATE, DIVERSITY_THRESHOLD, DIVERSITY_ACCEPTANCE_PROB
from tchebycheff_utils import tchebycheff_eval

import numpy as np
import random
from copy import deepcopy

def evaluate_individual(ind, projects, delta_matrix, n_samples=10):
    import numpy as np

    Z1, Z2, Z3 = 0.0, 0.0, 0.0
    n = len(projects)

    benefit_lambda = {
        'Operational Efficiency': 1.50,
        'Customer Experience': 1.43,
        'Business Culture': 1.00
    }

    for i in range(n):
        if ind['x'][i] == 1:
            p = projects[i]

            # λ_b based on benefit group
            λ_b = benefit_lambda.get(p.get('benefit_group', 'Business Culture'), 1.00)

            # Strategic value including synergy contribution
            synergy_score = p['synergy_same'] + p['synergy_cross']
            Z1 += (p['svs'] + λ_b * synergy_score) * (1 - p['risk'])

            # === Monte Carlo fuzzy cost sampling ===
            c1, c2, c3 = p['fuzzy_cost']
            fuzzy_samples = np.random.triangular(c1, c2, c3, n_samples)

            # Funding cost multiplier based on funding mix
            funding_penalty = (
                p['alpha'] * 0.9 +
                p['beta'] * 1.0 +
                p['theta'] * 1.3 +
                p['gamma'] * 1.1 +
                p['delta'] * 1.2
            )

            total_effective_cost = 0.0
            for sample_cost in fuzzy_samples:
                base_cost = max(1.0, sample_cost - synergy_score)
                effective_cost = base_cost * funding_penalty
                effective_cost = min(effective_cost, base_cost * 1.5)
                total_effective_cost += effective_cost

            avg_cost = total_effective_cost / n_samples
            Z2 += avg_cost * p['risk']
            Z3 += synergy_score

    # Z1: inter-project synergy weighted by avg λ from both projects
    for i in range(n):
        for j in range(i + 1, n):
            if ind['x'][i] == 1 and ind['x'][j] == 1:
                λ_i = benefit_lambda.get(projects[i].get('benefit_group', 'Business Culture'), 1.00)
                λ_j = benefit_lambda.get(projects[j].get('benefit_group', 'Business Culture'), 1.00)
                avg_λ = (λ_i + λ_j) / 2
                Z1 += avg_λ * delta_matrix[i][j]

    return [Z1, Z2, Z3]

def update_ideal_point(Z, ideal):
    ideal[0] = max(ideal[0], Z[0])
    ideal[1] = min(ideal[1], Z[1])
    ideal[2] = max(ideal[2], Z[2])

import numpy as np
import random
import itertools

import numpy as np
import random
import itertools

def initialize_ifpom(pop_size, num_projects, num_neighbors=20):
    population = []

    for _ in range(pop_size):
        # Inisialisasi x (keputusan proyek)
        x = [1 if random.random() < 0.5 else 0 for _ in range(num_projects)]
        if sum(x) == 0:
            x[random.randint(0, num_projects - 1)] = 1

        # Inisialisasi funding (α–δ) per proyek
        funding = []
        for _ in range(num_projects):
            f = np.random.dirichlet([1, 1, 1, 1, 1])
            f[2] = min(f[2], 0.4)  # batas vendor financing θ
            funding.append(f)

        funding = np.array(funding)
        ind = {
            'x': x,
            'alpha': funding[:, 0].tolist(),
            'beta': funding[:, 1].tolist(),
            'theta': funding[:, 2].tolist(),
            'gamma': funding[:, 3].tolist(),
            'delta': funding[:, 4].tolist(),
            'Z': [None, None, None]
        }
        population.append(ind)

    # Fungsi bantu: generate vektor bobot terdistribusi merata
    def uniform_weight_vectors(n_objs, divisions):
        vectors = []
        for partition in itertools.combinations_with_replacement(range(divisions + 1), n_objs):
            if sum(partition) == divisions:
                vector = np.array(partition) / divisions
                vectors.append(vector)
        return np.array(vectors)

    # Generate dan pastikan jumlah weight_vectors = pop_size
    raw_vectors = uniform_weight_vectors(n_objs=3, divisions=13)  # bisa menghasilkan > 100
    if len(raw_vectors) < pop_size:
        # Jika kurang, tambahkan vektor acak
        extra = np.random.dirichlet(np.ones(3), size=pop_size - len(raw_vectors))
        weight_vectors = np.vstack([raw_vectors, extra])
    else:
        weight_vectors = raw_vectors[:pop_size]

    # Hitung jarak dan neighborhood
    distances = np.linalg.norm(weight_vectors[:, None, :] - weight_vectors[None, :, :], axis=2)
    neighborhoods = np.argsort(distances, axis=1)[:, :min(num_neighbors, pop_size)]

    # Validasi akhir
    assert len(population) == len(weight_vectors) == len(neighborhoods), \
        f"Mismatch: population={len(population)}, weights={len(weight_vectors)}, neighbors={len(neighborhoods)}"

    return population, weight_vectors, neighborhoods


def crossover(parent1, parent2):
    child = {'x': [], 'alpha': [], 'beta': [], 'theta': [], 'gamma': [], 'delta': [], 'Z': [None, None, None]}
    for key in ['x', 'alpha', 'beta', 'theta', 'gamma', 'delta']:
        for i in range(len(parent1[key])):
            val = parent1[key][i] if random.random() < 0.5 else parent2[key][i]
            child[key].append(val)
    return child

from copy import deepcopy

def moead_generation(
    population, projects, delta_matrix,
    weight_vectors, neighborhoods,
    ideal_point, gen, max_gen,
    z_min, z_max
):
    mutation_prob = INITIAL_MUTATION_RATE * (1 - gen / max_gen) + MIN_MUTATION_RATE

    for i in range(len(population)):
        neighbors = neighborhoods[i]
        p1, p2 = random.sample(list(neighbors), 2)

        # --- Crossover
        child = crossover(population[p1], population[p2])

        # --- Mutasi variabel keputusan proyek (x)
        for m in range(len(projects)):
            if random.random() < mutation_prob or child['x'][m] == population[p1]['x'][m]:
                child['x'][m] = 1 - child['x'][m]

        # --- Pastikan minimal 1 proyek dipilih
        if sum(child['x']) == 0:
            child['x'][random.randint(0, len(projects) - 1)] = 1

        # --- Mutasi rasio pembiayaan
        for m in range(len(projects)):
            if child['x'][m] == 1 and random.random() < mutation_prob:
                ratios = np.array([
                    child['alpha'][m],
                    child['beta'][m],
                    child['theta'][m],
                    child['gamma'][m],
                    child['delta'][m]
                ])
                noise = np.random.normal(0, 0.02, size=5)
                mutated = ratios + noise
                mutated = np.maximum(mutated, 0.0001)
                mutated /= mutated.sum()

                # Batas theta ≤ 0.4
                if mutated[2] > 0.4:
                    excess = mutated[2] - 0.4
                    mutated[2] = 0.4
                    redistribute_idx = [0, 1, 3, 4]
                    redistributed = mutated[redistribute_idx] + (
                        excess * mutated[redistribute_idx] / mutated[redistribute_idx].sum()
                    )
                    for j, idx in enumerate(redistribute_idx):
                        mutated[idx] = redistributed[j]

                # Simpan kembali
                child['alpha'][m] = mutated[0]
                child['beta'][m] = mutated[1]
                child['theta'][m] = mutated[2]
                child['gamma'][m] = mutated[3]
                child['delta'][m] = mutated[4]

        # --- Evaluasi anak
        child['Z'] = evaluate_individual(child, projects, delta_matrix)

        # --- Seleksi terhadap tetangga
        for j in neighbors:
            old_fit = tchebycheff_eval(population[j]['Z'], weight_vectors[j], ideal_point, z_min, z_max)
            new_fit = tchebycheff_eval(child['Z'], weight_vectors[j], ideal_point, z_min, z_max)
            diversity_x = sum(child['x'][k] != population[j]['x'][k] for k in range(len(child['x'])))

            if new_fit < old_fit or (diversity_x >= DIVERSITY_THRESHOLD and random.random() < DIVERSITY_ACCEPTANCE_PROB):
                population[j] = deepcopy(child)

        # --- Update ideal point dan batas normalisasi
        update_ideal_point(child['Z'], ideal_point)
        for k in range(3):
            z_min[k] = min(z_min[k], child['Z'][k])
            z_max[k] = max(z_max[k], child['Z'][k])

    return population, ideal_point, z_min, z_max

