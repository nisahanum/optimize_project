# Step 1: Definisi Lingkungan Proyek (Python)
#  1.1. Parameter Umum dan Import
import numpy as np
import random

# Jumlah proyek dalam portofolio
NUM_PROJECTS = 10

# Biaya modal masing-masing pendanaan
r_E = 0.0   # Internal Equity
r_S = 0.03  # Soft Loan
r_V = 0.08  # Vendor Financing
r_G = 0.01  # Grant/Subsidi
r_D = 0.06  # PPP / Joint Venture


# 1.2. Representasi Dataset Proyek

# Contoh dataset simulatif untuk 10 proyek
projects = []

for i in range(NUM_PROJECTS):
    project = {
        'id': f'P{i+1}',
        'classification': random.choice(['Operational Efficiency', 'Customer Experience', 'Business Culture']),
        'svs': random.uniform(60, 90),  # Strategic Value Score
        'trl': random.randint(1, 9),    # TRL level
        'complexity': round(random.uniform(0.3, 0.9), 2),  # Kompleksitas
        'fuzzy_cost': (random.randint(1000, 2000), random.randint(2000, 3000), random.randint(3000, 4000)),
        'alpha': round(random.uniform(0.1, 0.6), 2),
        'beta': 0.0,  # akan dinormalisasi nanti
        'theta': 0.0,
        'gamma': 0.0,
        'delta': 0.0,
        'synergy_same': round(random.uniform(0, 100), 2),
        'synergy_cross': round(random.uniform(0, 100), 2)
    }
    
    # Normalisasi rasio pendanaan agar total = 1
    remaining = 1.0 - project['alpha']
    shares = np.random.dirichlet(np.ones(4), size=1)[0]
    project['beta'], project['theta'], project['gamma'], project['delta'] = (remaining * s for s in shares)
    
    projects.append(project)
    
# 1.3. Fungsi Risiko Teknis dan Finansial


def compute_technical_risk(trl, complexity):
    return ((9 - trl) / 8) * complexity

def compute_financial_risk(project):
    return (
        project['alpha'] * 0.0 +
        project['beta'] * 0.3 +
        project['theta'] * 1.0 +
        project['gamma'] * 0.1 +
        project['delta'] * 0.6
    )

# 1.4. Menyimpan Risiko dan Nilai Akhir ke Proyek
for p in projects:
    p['risk_tech'] = compute_technical_risk(p['trl'], p['complexity'])
    p['risk_fin'] = compute_financial_risk(p)
    p['risk'] = max(0.05, 0.6 * p['risk_tech'] + 0.4 * p['risk_fin'])  # ⬅️ Tambahan pengaman


# Contoh Output untuk 1 Proyek
#for p in projects[:1]:
#    print(p)

#Step 2: Perhitungan Fungsi Objektif IFPOM
#2.1. Fungsi Z₁ – Adjusted Strategic Value + Synergy Pairwise

def compute_Z1(projects, x, delta_matrix, lambda_val=1.0):
    z1 = 0.0
    for i, p in enumerate(projects):
        if x[i] == 1:
            z1 += p['svs'] * (1 - p['risk'])

    # Pairwise synergy antara proyek-proyek terpilih
    for i in range(len(projects)):
        for j in range(i+1, len(projects)):
            if x[i] == 1 and x[j] == 1:
                z1 += lambda_val * delta_matrix[i][j]
    
    return z1

#2.2. Fungsi Z₂ – Risk-Adjusted Financial Cost (dengan Fuzzy Cost dan Pendanaan 5 Tipe)

def expected_fuzzy_cost(fuzzy_tuple):
    c_min, c_likely, c_max = fuzzy_tuple
    return (c_min + 2 * c_likely + c_max) / 4

def compute_Z2(projects, x, r_E, r_S, r_V, r_G, r_D):
    z2 = 0.0
    for i, p in enumerate(projects):
        if x[i] == 1:
            # Langkah 1: Hitung ekspektasi biaya fuzzy
            raw_cost = expected_fuzzy_cost(p['fuzzy_cost'])

            # Langkah 2: Hitung sinergi gabungan
            synergy_total = p['synergy_same'] + p['synergy_cross']

            # ✅ Perbaikan: Jangan izinkan biaya menjadi negatif
            adjusted_cost = max(1.0, raw_cost - synergy_total)

            # Langkah 3: Hitung bobot biaya pendanaan
            funding_weight = (
                p['alpha'] * r_E +
                p['beta'] * r_S +
                p['theta'] * r_V +
                p['gamma'] * r_G +
                p['delta'] * r_D
            )

            # Langkah 4: Hitung total Z2 kontribusi proyek
            z2 += adjusted_cost * funding_weight * p['risk']

    return z2


#2.3. Fungsi Z₃ – Total Synergy (Same-Period + Cross-Period)

def compute_Z3(projects, x):
    z3 = 0.0
    for i, p in enumerate(projects):
        if x[i] == 1:
            z3 += p['synergy_same'] + p['synergy_cross']
    return z3

# 2.4. Contoh Evaluasi Individu

# Dummy decision vector (x): memilih 1 jika indeks genap
x_sample = [1 if i % 2 == 0 else 0 for i in range(NUM_PROJECTS)]

# Dummy matrix delta[i][j] – nilai sinergi pasangan proyek (pairwise)
delta_matrix = np.random.uniform(0, 50, size=(NUM_PROJECTS, NUM_PROJECTS))
np.fill_diagonal(delta_matrix, 0)

# Evaluasi
z1 = compute_Z1(projects, x_sample, delta_matrix)
z2 = compute_Z2(projects, x_sample, r_E, r_S, r_V, r_G, r_D)
z3 = compute_Z3(projects, x_sample)

print(f'Z1 = {z1:.2f}, Z2 = {z2:.2f}, Z3 = {z3:.2f}')

#  Step 3: Inisialisasi Populasi dan Tetangga untuk MOEA/D

# 3.1. Parameter Dasar MOEA/D

POP_SIZE = 50          # Jumlah individu dalam populasi
NUM_OBJECTIVES = 3     # Z1, Z2, Z3
T = 10                 # Ukuran tetangga

# 3.2. Membuat Vektor Bobot Tujuan

def generate_weight_vectors(pop_size, num_objectives):
    return np.random.dirichlet(np.ones(num_objectives), size=pop_size)

weight_vectors = generate_weight_vectors(POP_SIZE, NUM_OBJECTIVES)

# 3.3. Menentukan Tetangga Berdasarkan Kedekatan Bobot

def calculate_neighborhoods(weights, T):
    distances = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=2)
    return np.argsort(distances, axis=1)[:, :T]

neighborhoods = calculate_neighborhoods(weight_vectors, T)

# 3.4. Inisialisasi Populasi Awal

def initialize_population(pop_size, num_projects):
    population = []
    for _ in range(pop_size):
        x = np.random.randint(0, 2, num_projects).tolist()

        # Random mix of 5 funding types (alpha–delta) per proyek
        funding_mix = [np.random.dirichlet([1, 1, 1, 1, 1]).tolist() for _ in range(num_projects)]

        individual = {
            'x': x,
            'alpha': [f[0] for f in funding_mix],
            'beta':  [f[1] for f in funding_mix],
            'theta': [f[2] for f in funding_mix],
            'gamma': [f[3] for f in funding_mix],
            'delta': [f[4] for f in funding_mix],
            'Z': [None, None, None]
        }
        population.append(individual)
    return population

population = initialize_population(POP_SIZE, NUM_PROJECTS)

# 3.5. Output Sample Individu

print(population[0])

# Step 4: Loop MOEA/D — Evaluasi, Update Tetangga, dan Evolusi Generasi

# 4.1. Setup Ideal Point dan Evaluasi Awal

ideal_point = [float('inf')] * 3  # Z1, Z2, Z3

def evaluate_individual(ind, projects, delta_matrix):
    Z1 = compute_Z1(projects, ind['x'], delta_matrix)
    Z2 = compute_Z2(projects, ind['x'], r_E, r_S, r_V, r_G, r_D)
    Z3 = compute_Z3(projects, ind['x'])
    return [Z1, Z2, Z3]

# 4.2. Evaluasi dan Update Ideal Point

for ind in population:
    ind['Z'] = evaluate_individual(ind, projects, delta_matrix)
    for i in range(3):
        if ind['Z'][i] < ideal_point[i]:
            ideal_point[i] = ind['Z'][i]
            
# 4.3. Fungsi Evaluasi Tchebycheff
def tchebycheff_eval(Z, weight, ideal):
    return max([weight[i] * abs(Z[i] - ideal[i]) for i in range(len(Z))])

# 4.4. Crossover + Mutasi + Update Tetangga

def crossover(parent1, parent2):
    child = {'x': [], 'alpha': [], 'beta': [], 'theta': [], 'gamma': [], 'delta': []}
    for key in ['x', 'alpha', 'beta', 'theta', 'gamma', 'delta']:
        for i in range(NUM_PROJECTS):
            if random.random() < 0.5:
                child[key].append(parent1[key][i])
            else:
                child[key].append(parent2[key][i])
    return child

# 4.5. MOEA/D Main Loop

# === STEP 4.4: MOEA/D MAIN LOOP ===
NUM_GENERATIONS = 100

for gen in range(NUM_GENERATIONS):
    for i in range(POP_SIZE):
        neighbors = neighborhoods[i]
        p1, p2 = random.sample(list(neighbors), 2)

        # === Crossover ===
        child = crossover(population[p1], population[p2])

        # === Mutasi Ringan ===
        for m in range(NUM_PROJECTS):
            if random.random() < 0.1:
                child['x'][m] = 1 - child['x'][m]

        # Pastikan tidak semua proyek dinonaktifkan
        if sum(child['x']) == 0:
            child['x'][random.randint(0, NUM_PROJECTS - 1)] = 1

        # === Evaluasi Fungsi Objektif ===
        child['Z'] = evaluate_individual(child, projects, delta_matrix)

        # === Update Tetangga jika child lebih baik (Tchebycheff-based) ===
        for j in neighborhoods[i]:
            old_fitness = tchebycheff_eval(population[j]['Z'], weight_vectors[j], ideal_point)
            new_fitness = tchebycheff_eval(child['Z'], weight_vectors[j], ideal_point)

            if new_fitness < old_fitness:
                # Simpan salinan child ke tetangga
                population[j] = {
                    'x': child['x'][:],
                    'alpha': child['alpha'][:],
                    'beta': child['beta'][:],
                    'theta': child['theta'][:],
                    'gamma': child['gamma'][:],
                    'delta': child['delta'][:],
                    'Z': child['Z'][:]
                }

        # === Update Ideal Point Sesuai Arah Objektif ===
        if child['Z'][0] > ideal_point[0]:
            ideal_point[0] = child['Z'][0]  # Z1 max
        if child['Z'][1] < ideal_point[1]:
            ideal_point[1] = child['Z'][1]  # Z2 min
        if child['Z'][2] > ideal_point[2]:
            ideal_point[2] = child['Z'][2]  # Z3 max

    # Cetak hasil setiap 10 generasi
    if gen % 10 == 0:
        print(f"Generation {gen}: Best Z1 = {ideal_point[0]:.2f}, Z2 = {ideal_point[1]:.2f}, Z3 = {ideal_point[2]:.2f}")


# Step 5: Visualisasi Pareto Front (Z₁, Z₂, Z₃)

# 5.1 Import dan Setup Visualisasi

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 5.2 Ekstrak Nilai Objektif dari Populasi Akhir

Z1_vals = [ind['Z'][0] for ind in population if ind['Z'] is not None]
Z2_vals = [ind['Z'][1] for ind in population if ind['Z'] is not None]
Z3_vals = [ind['Z'][2] for ind in population if ind['Z'] is not None]

#  5.3 Visualisasi 3D Pareto Front
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(Z1_vals, Z2_vals, Z3_vals, c='blue', marker='o')

ax.set_xlabel('Z1 - Adjusted Strategic Value')
ax.set_ylabel('Z2 - Risk-Adjusted Financial Cost')
ax.set_zlabel('Z3 - Total Synergy')
plt.title('Pareto Front - IFPOM Optimization')
plt.grid(True)
plt.show()

# (Opsional) Label Titik atau Highlight Solusi Terbaik
best_idx = Z1_vals.index(max(Z1_vals))  # solusi dengan nilai strategis tertinggi
ax.scatter(Z1_vals[best_idx], Z2_vals[best_idx], Z3_vals[best_idx], c='red', marker='^', s=100)

