import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Konfigurasi Simulasi
n_projects = 10
pop_size = 50
max_gen = 100
lambda_val = 0.05

# Fungsi untuk Biaya Fuzzy
def expected_fuzzy_cost(cost):
    c_min, c_likely, c_max = cost
    return (c_min + 2 * c_likely + c_max) / 4

# Generate Data Dummy dengan Skala Ketidakpastian Fleksibel
def generate_data(market_range=(0.5, 1.0), reg_range=(0.0, 0.3)):
    data = {
        'svs': np.random.uniform(50, 100, n_projects),
        'risk_fin': np.random.uniform(0.1, 0.4, n_projects),
        'risk_tech': np.random.uniform(0.1, 0.4, n_projects),
        'fuzzy_cost': [(random.uniform(800, 1000), random.uniform(1000, 1200), random.uniform(1200, 1500)) for _ in range(n_projects)],
        'synergy': np.random.uniform(50, 150, n_projects),
        'synergy_same': np.random.uniform(10, 100, n_projects),
        'synergy_cross': np.random.uniform(10, 100, n_projects),
        'delta': np.random.uniform(0, 1, (n_projects, n_projects)),
        'r_E': 0.0,
        'r_S': 0.03,
        'r_V': 0.08,
        'lambda': lambda_val,
        'market_adoption': np.random.uniform(*market_range, n_projects),
        'regulatory_impact': np.random.uniform(*reg_range, n_projects)
    }
    return data

# Inisialisasi Individu
def create_individual():
    x = [random.randint(0, 1) for _ in range(n_projects)]
    alpha = [random.uniform(0, 1) for _ in range(n_projects)]
    beta = [random.uniform(0, 1 - alpha[i]) for i in range(n_projects)]
    theta = [1 - alpha[i] - beta[i] for i in range(n_projects)]
    for i in range(n_projects):
        if theta[i] > 0.4:
            excess = theta[i] - 0.4
            theta[i] = 0.4
            alpha[i] += excess
    return {'x': x, 'alpha': alpha, 'beta': beta, 'theta': theta, 'Z': [None]*3}

# Evaluasi Fungsi Objektif
def evaluate_individual(ind, data):
    x, alpha, beta, theta = ind['x'], ind['alpha'], ind['beta'], ind['theta']
    Z1 = sum(data['svs'][i] * (1 - data['risk_tech'][i]) * data['market_adoption'][i] * x[i] for i in range(n_projects))
    Z1 += data['lambda'] * sum(data['delta'][i][j] * x[i] * x[j] for i in range(n_projects) for j in range(i+1, n_projects))

    Z2 = sum(((expected_fuzzy_cost(data['fuzzy_cost'][i]) - data['synergy'][i]) *
              (alpha[i] * data['r_E'] + beta[i] * data['r_S'] + theta[i] * data['r_V']) *
              (data['risk_fin'][i] + data['regulatory_impact'][i]) * x[i]) for i in range(n_projects))

    Z3 = sum((data['synergy_same'][i] + data['synergy_cross'][i]) * x[i] for i in range(n_projects))
    ind['Z'] = [Z1, Z2, Z3]

# Simulasi Generasi
def simulate(data):
    population = [create_individual() for _ in range(pop_size)]
    for ind in population:
        evaluate_individual(ind, data)

    for _ in range(max_gen):
        new_pop = []
        for _ in range(pop_size):
            p1, p2 = random.sample(population, 2)
            child = crossover(p1, p2)
            mutate(child)
            evaluate_individual(child, data)
            new_pop.append(child)
        population = new_pop
    return population

# Crossover dan Mutasi
def crossover(p1, p2):
    child = {'x': [], 'alpha': [], 'beta': [], 'theta': [], 'Z': [None]*3}
    for i in range(n_projects):
        child['x'].append(random.choice([p1['x'][i], p2['x'][i]]))
        a = (p1['alpha'][i] + p2['alpha'][i]) / 2
        b = (p1['beta'][i] + p2['beta'][i]) / 2
        t = 1 - a - b
        t = min(t, 0.4)
        if a + b + t < 1:
            a += 1 - (a + b + t)
        child['alpha'].append(a)
        child['beta'].append(b)
        child['theta'].append(t)
    return child

def mutate(ind, rate=0.1):
    for i in range(n_projects):
        if random.random() < rate:
            ind['x'][i] = 1 - ind['x'][i]
        if random.random() < rate:
            a = random.uniform(0, 1)
            b = random.uniform(0, 1 - a)
            t = 1 - a - b
            if t > 0.4: t = 0.4
            ind['alpha'][i] = a
            ind['beta'][i] = b
            ind['theta'][i] = t

# Visualisasi Pareto
def plot_pareto(pop, label):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    z1 = [i['Z'][0] for i in pop]
    z2 = [i['Z'][1] for i in pop]
    z3 = [i['Z'][2] for i in pop]
    ax.scatter(z1, z2, z3, label=label, alpha=0.6)
    ax.set_xlabel('Z1 - Adjusted Strategic Value')
    ax.set_ylabel('Z2 - Risk-Adjusted Financial Cost')
    ax.set_zlabel('Z3 - Total Synergy')
    ax.set_title(f'Pareto Front - {label}')
    ax.legend()
    plt.show()

# Simulasi untuk dua kondisi: best-case vs worst-case uncertainty
data_best = generate_data(market_range=(0.8, 1.0), reg_range=(0.0, 0.1))
data_worst = generate_data(market_range=(0.2, 0.5), reg_range=(0.3, 0.6))

pop_best = simulate(data_best)
pop_worst = simulate(data_worst)

# Visualisasi
plot_pareto(pop_best, "Best-Case Uncertainty")
plot_pareto(pop_worst, "Worst-Case Uncertainty")
