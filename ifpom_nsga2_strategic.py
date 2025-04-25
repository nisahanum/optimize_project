import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# Konfigurasi masalah
NUM_PROJECTS = 20
LAMBDA_SYNERGY = 0.05
MAX_SELECTED = 10

# Setup DEAP
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)
toolbox = base.Toolbox()

# Data proyek dummy
project_data = [{
    'roi': random.uniform(10, 100),
    'cost': random.uniform(1000, 10000),
    'synergy': random.uniform(50, 500),
    'risk': random.uniform(0.05, 0.4),
} for _ in range(NUM_PROJECTS)]

# Inisialisasi individu dengan x_p random
def init_individual():
    ind = []
    for _ in range(NUM_PROJECTS):
        x = 1 if random.random() < 0.3 else 0
        a = random.uniform(0, 1)
        b = random.uniform(0, 1 - a)
        c = 1 - a - b
        s = random.randint(1, 6)
        ind += [x, a, b, c, s]
    return ind

toolbox.register("individual", tools.initIterate, creator.Individual, init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Fungsi evaluasi dengan penalti jumlah proyek
def evaluate(individual):
    total_roi = 0
    total_risk_adjusted_cost = 0
    total_synergy = 0
    selected = 0

    for i in range(NUM_PROJECTS):
        x = int(round(individual[i * 5]))
        a = individual[i * 5 + 1]
        b = individual[i * 5 + 2]
        c = individual[i * 5 + 3]
        s = individual[i * 5 + 4]

        if x == 0:
            continue

        selected += 1
        proj = project_data[i]
        ROI = proj['roi']
        cost = proj['cost']
        risk = proj['risk']
        synergy = proj['synergy']

        total_roi += ROI * (1 - risk)
        total_synergy += synergy
        total_risk_adjusted_cost += (a + b + c) * cost * (1 + risk)

    if selected > MAX_SELECTED:
        penalty = sum([project_data[i]['cost'] for i in range(NUM_PROJECTS) if int(round(individual[i*5])) == 1])
        total_risk_adjusted_cost += penalty * 0.1

    strategic_value = total_roi + LAMBDA_SYNERGY * total_synergy
    return -strategic_value, total_risk_adjusted_cost

toolbox.register("evaluate", evaluate)

# Mutasi dengan binary flipping dan normalisasi proporsi
def custom_mutate(individual):
    for i in range(0, len(individual), 5):
        if random.random() < 0.1:
            individual[i] = 1 - int(round(individual[i]))  # flip x_p
        a = individual[i + 1]
        b = individual[i + 2]
        c = individual[i + 3]
        total = a + b + c
        if total == 0:
            a, b, c = 1.0, 0.0, 0.0
        else:
            a, b, c = a/total, b/total, c/total
        individual[i + 1] = a
        individual[i + 2] = b
        individual[i + 3] = c
    return individual,

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", custom_mutate)
toolbox.register("select", tools.selNSGA2)

# Algoritma utama
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

# Eksekusi
final_pop, logbook, pareto = main()

# Visualisasi Pareto Front
f1 = [-ind.fitness.values[0] for ind in pareto]
f2 = [ind.fitness.values[1] for ind in pareto]

plt.figure(figsize=(8, 6))
plt.scatter(f2, f1, c="green")
plt.xlabel("Risk-Adjusted Cost (Z2)")
plt.ylabel("Strategic Value: ROI + λ·Synergy (Z1')")
plt.title("Pareto Front after Penalizing Over-Selection")
plt.grid()
plt.show()

# Debug Pareto isi
print("=== Pareto Front Individuals ===")
for i, ind in enumerate(pareto):
    print(f"Individu {i+1}:")
    print(f"  Fitness (Strategic Value, Cost): {ind.fitness.values}")
    print(f"  Chromosome: {ind}")
    print("-" * 50)
