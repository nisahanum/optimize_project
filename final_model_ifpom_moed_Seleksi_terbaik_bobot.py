# MOEA/D untuk Model IFPOM (Simulasi Awal)
# Penulis: [Nama Anda]
# Deskripsi: Implementasi awal struktur algoritma MOEA/D untuk simulasi optimasi portofolio proyek IFPOM

import numpy as np
import random
import matplotlib.pyplot as plt

# -------------------------------
# Konfigurasi Awal
# -------------------------------
n_projects = 10              # Jumlah proyek simulasi
pop_size = 50                # Ukuran populasi (jumlah solusi)
n_objectives = 3            # Z1, Z2, Z3
lambda_val = 0.05           # Koefisien sinergi di Z1
max_gen = 100               # Jumlah generasi evolusi

# -------------------------------
# Data Simulatif Proyek
# -------------------------------
def generate_dummy_data():
    data = {
        'svs': np.random.uniform(50, 100, n_projects),
        'risk_fin': np.random.uniform(0.1, 0.4, n_projects),
        'risk_tech': np.random.uniform(0.1, 0.4, n_projects),
        'fuzzy_cost': [
            (random.uniform(800, 1000), random.uniform(1000, 1200), random.uniform(1200, 1500))
            for _ in range(n_projects)
        ],
        'synergy': np.random.uniform(50, 150, n_projects),
        'synergy_same': np.random.uniform(10, 100, n_projects),
        'synergy_cross': np.random.uniform(10, 100, n_projects),
        'delta': np.random.uniform(0, 1, (n_projects, n_projects)),
        'r_E': 0.0,
        'r_S': 0.03,
        'r_V': 0.08,
        'lambda': lambda_val
    }
    return data

# -------------------------------
# Fungsi Membuat Individu
# -------------------------------
def create_individual():
    x = [random.randint(0, 1) for _ in range(n_projects)]
    alpha = [random.uniform(0, 1) for _ in range(n_projects)]
    beta = [random.uniform(0, 1 - alpha[i]) for i in range(n_projects)]
    theta = [1 - alpha[i] - beta[i] for i in range(n_projects)]
    for i in range(n_projects):
        if theta[i] > 0.4:
            excess = theta[i] - 0.4
            theta[i] = 0.4
            alpha[i] += excess  # redistribusi
    return {'x': x, 'alpha': alpha, 'beta': beta, 'theta': theta, 'Z': [None]*n_objectives}

# -------------------------------
# Evaluasi Fungsi Objektif
# -------------------------------
def expected_fuzzy_cost(cost):
    c_min, c_likely, c_max = cost
    return (c_min + 2 * c_likely + c_max) / 4

def evaluate_individual(ind, data):
    x, alpha, beta, theta = ind['x'], ind['alpha'], ind['beta'], ind['theta']
    risk_fin = data['risk_fin']
    risk_tech = data['risk_tech']

    Z1 = sum(data['svs'][i] * (1 - risk_tech[i]) * x[i] for i in range(n_projects))
    Z1 += data['lambda'] * sum(data['delta'][i][j] * x[i] * x[j] for i in range(n_projects) for j in range(i+1, n_projects))

    Z2 = sum(((expected_fuzzy_cost(data['fuzzy_cost'][i]) - data['synergy'][i]) *
              (alpha[i] * data['r_E'] + beta[i] * data['r_S'] + theta[i] * data['r_V']) *
              risk_fin[i] * x[i]) for i in range(n_projects))

    Z3 = sum((data['synergy_same'][i] + data['synergy_cross'][i]) * x[i] for i in range(n_projects))
    ind['Z'] = [Z1, Z2, Z3]

# -------------------------------
# Operator Evolusi
# -------------------------------
def crossover(parent1, parent2):
    child = {'x': [], 'alpha': [], 'beta': [], 'theta': [], 'Z': [None]*n_objectives}
    for i in range(n_projects):
        child['x'].append(random.choice([parent1['x'][i], parent2['x'][i]]))
        a = (parent1['alpha'][i] + parent2['alpha'][i]) / 2
        b = (parent1['beta'][i] + parent2['beta'][i]) / 2
        t = 1 - a - b
        t = min(t, 0.4)
        if a + b + t < 1:
            a += 1 - (a + b + t)
        child['alpha'].append(a)
        child['beta'].append(b)
        child['theta'].append(t)
    return child

def mutate(ind, mutation_rate=0.1):
    for i in range(n_projects):
        if random.random() < mutation_rate:
            ind['x'][i] = 1 - ind['x'][i]
        if random.random() < mutation_rate:
            a = random.uniform(0, 1)
            b = random.uniform(0, 1 - a)
            t = 1 - a - b
            if t > 0.4: t = 0.4
            ind['alpha'][i] = a
            ind['beta'][i] = b
            ind['theta'][i] = t

# -------------------------------
# Visualisasi Pareto Front
# -------------------------------
def plot_pareto(population):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Z1_vals = [ind['Z'][0] for ind in population]
    Z2_vals = [ind['Z'][1] for ind in population]
    Z3_vals = [ind['Z'][2] for ind in population]
    ax.scatter(Z1_vals, Z2_vals, Z3_vals, c='blue', marker='o')
    ax.set_xlabel('Z1 - Adjusted Strategic Value')
    ax.set_ylabel('Z2 - Risk-Adjusted Financial Cost')
    ax.set_zlabel('Z3 - Total Synergy')
    plt.title('Pareto Front - IFPOM Simulation')
    plt.show()

# -------------------------------
# Visualisasi Konvergensi 2D
# -------------------------------
def plot_convergence(logbook):
    gens = list(range(len(logbook['Z1'])))
    plt.figure()
    plt.plot(gens, logbook['Z1'], label='Z1', color='blue')
    plt.plot(gens, logbook['Z2'], label='Z2', color='orange')
    plt.plot(gens, logbook['Z3'], label='Z3', color='green')
    plt.xlabel('Generation')
    plt.ylabel('Average Objective Value')
    plt.title('Convergence Over Generations')
    plt.legend()
    plt.grid(True)
    plt.show()

# -------------------------------
# Seleksi Solusi Terbaik (Weighted Preference)
# -------------------------------
def select_best_solution(population, weights=(0.4, 0.3, 0.3)):
    Z1_vals = np.array([ind['Z'][0] for ind in population])
    Z2_vals = np.array([ind['Z'][1] for ind in population])
    Z3_vals = np.array([ind['Z'][2] for ind in population])

    norm_Z1 = (Z1_vals - Z1_vals.min()) / (Z1_vals.max() - Z1_vals.min())
    norm_Z2 = (Z2_vals - Z2_vals.min()) / (Z2_vals.max() - Z2_vals.min())
    norm_Z3 = (Z3_vals - Z3_vals.min()) / (Z3_vals.max() - Z3_vals.min())

    scores = weights[0] * norm_Z1 + weights[1] * (1 - norm_Z2) + weights[2] * norm_Z3
    best_idx = np.argmax(scores)
    best_individual = population[best_idx]
    print(f"Best solution (based on weighted preference):\nZ1={best_individual['Z'][0]:.2f}, Z2={best_individual['Z'][1]:.2f}, Z3={best_individual['Z'][2]:.2f}")
    return best_individual

# -------------------------------
# Simulasi MOEA/D Sederhana
# -------------------------------
def run_initial_simulation():
    data = generate_dummy_data()
    population = [create_individual() for _ in range(pop_size)]
    for ind in population:
        evaluate_individual(ind, data)

    logbook = {'Z1': [], 'Z2': [], 'Z3': []}

    for gen in range(max_gen):
        new_population = []
        for _ in range(pop_size):
            p1, p2 = random.sample(population, 2)
            child = crossover(p1, p2)
            mutate(child)
            evaluate_individual(child, data)
            new_population.append(child)
        population = new_population

        avg_z1 = np.mean([ind['Z'][0] for ind in population])
        avg_z2 = np.mean([ind['Z'][1] for ind in population])
        avg_z3 = np.mean([ind['Z'][2] for ind in population])
        logbook['Z1'].append(avg_z1)
        logbook['Z2'].append(avg_z2)
        logbook['Z3'].append(avg_z3)

    plot_pareto(population)
    plot_convergence(logbook)
    select_best_solution(population)

if __name__ == "__main__":
    run_initial_simulation()
