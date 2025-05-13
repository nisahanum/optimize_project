import random
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

# Konfigurasi masalah
NUM_PROJECTS = 10
max_project_limit = 6
penalty_weight = 500

# Buat struktur fitness dan individual
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))  # Minimize -Z1, Minimize Z2
creator.create("Individual", list, fitness=creator.FitnessMulti)
toolbox = base.Toolbox()

# Data dummy proyek dengan variabilitas tinggi
project_data = [{
    'roi': random.uniform(10, 60),
    'cost': random.uniform(500, 3000),
    'synergy': random.uniform(10, 300),
    'risk': random.uniform(0.05, 0.4),
} for _ in range(NUM_PROJECTS)]

# Inisialisasi individual dengan normalisasi α + β + γ = 1
def init_individual():
    ind = []
    for _ in range(NUM_PROJECTS):
        x = random.randint(0, 1)
        a = random.uniform(0, 1)
        b = random.uniform(0, 1 - a)
        c = 1 - a - b
        s = random.randint(1, 6)
        ind += [x, a, b, c, s]
    return ind

toolbox.register("individual", tools.initIterate, creator.Individual, init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Fungsi evaluasi
def evaluate(individual):
    Z1 = 0
    Z2 = 0
    selected_projects = 0

    for i in range(NUM_PROJECTS):
        x = int(round(individual[i*5]))  # ensure binary
        a = individual[i*5 + 1]
        b = individual[i*5 + 2]
        c = individual[i*5 + 3]
        s = individual[i*5 + 4]

        if x == 0:
            continue

        selected_projects += 1
        proj = project_data[i]
        ROI = proj['roi']
        cost = proj['cost']
        synergy = proj['synergy']
        risk = proj['risk']

        adjusted_cost = max(cost - synergy, 0)
        risk_penalty = adjusted_cost * risk

        Z1 += ROI
        Z2 += (a + b + c) * adjusted_cost + risk_penalty

    if selected_projects > max_project_limit:
        Z2 += (selected_projects - max_project_limit) * penalty_weight

    return -Z1, Z2

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

# Perbaikan mutasi α, β, γ agar tetap valid
def custom_mutate(individual):
    tools.mutGaussian(individual, mu=0, sigma=0.2, indpb=0.2)
    for i in range(NUM_PROJECTS):
        a = individual[i*5 + 1]
        b = individual[i*5 + 2]
        c = individual[i*5 + 3]
        total = a + b + c
        if total == 0:
            a, b, c = 1.0, 0.0, 0.0
        else:
            a, b, c = a/total, b/total, c/total
        individual[i*5 + 1] = a
        individual[i*5 + 2] = b
        individual[i*5 + 3] = c
    return individual,

toolbox.register("mutate", custom_mutate)

# Main loop
def main():
    pop = toolbox.population(n=100)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    pop, log = algorithms.eaMuPlusLambda(
        pop, toolbox, mu=100, lambda_=200, cxpb=0.7, mutpb=0.3,
        ngen=50, stats=stats, halloffame=hof, verbose=True
    )

    return pop, log, hof

final_pop, logbook, pareto = main()

# Visualisasi hasil
f1 = [-ind.fitness.values[0] for ind in pareto]
f2 = [ind.fitness.values[1] for ind in pareto]

plt.scatter(f2, f1, c="red")
plt.xlabel("Risk-Informed Cost (Z2)")
plt.ylabel("Expected Portfolio Value (Z1)")
plt.title("Pareto Front from NSGA-II for IFPOM")
plt.grid()
plt.show()
