import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# ----------------------- Konfigurasi Masalah -----------------------

NUM_PROJECTS = 20

# Ubah ke 3 dimensi objektif
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0, -1.0))  # ROI(+), Cost(-), Synergy(+)
creator.create("Individual", list, fitness=creator.FitnessMulti)
toolbox = base.Toolbox()

# Data proyek dummy
project_data = [{
    'roi': random.uniform(10, 100),
    'cost': random.uniform(1000, 10000),
    'synergy': random.uniform(50, 500),
    'risk': random.uniform(0.05, 0.4),
} for _ in range(NUM_PROJECTS)]

# -------------------- Inisialisasi Kromosom ------------------------

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

# -------------------- Fungsi Evaluasi 3 Objektif ------------------------

def evaluate(individual):
    total_roi = 0
    total_risk_adjusted_cost = 0
    total_synergy = 0

    for i in range(NUM_PROJECTS):
        x = int(round(individual[i * 5]))  # binary project selection
        a = individual[i * 5 + 1]
        b = individual[i * 5 + 2]
        c = individual[i * 5 + 3]
        s = individual[i * 5 + 4]

        if x == 0:
            continue

        proj = project_data[i]
        ROI = proj['roi']
        cost = proj['cost']
        risk = proj['risk']
        synergy = proj['synergy']

        total_roi += ROI * (1 - risk)
        total_risk_adjusted_cost += (a + b + c) * cost * (1 + risk)
        total_synergy += synergy

    return -total_roi, total_risk_adjusted_cost, -total_synergy

toolbox.register("evaluate", evaluate)

# ------------------- Mutasi dan Crossover --------------------------

toolbox.register("mate", tools.cxTwoPoint)

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
toolbox.register("select", tools.selNSGA2)

# ------------------------ Algoritma Utama --------------------------

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

# ------------------------ Eksekusi & Visual ------------------------

if __name__ == "__main__":
    final_pop, logbook, pareto = main()

    f1 = [-ind.fitness.values[0] for ind in pareto]
    f2 = [ind.fitness.values[1] for ind in pareto]
    f3 = [-ind.fitness.values[2] for ind in pareto]

    # 2D plot Z1 vs Z2 (you can change axis to f3 if needed)
    plt.figure(figsize=(8, 6))
    plt.scatter(f2, f1, c="red")
    plt.xlabel("Risk-Adjusted Cost (Z2)")
    plt.ylabel("Adjusted ROI (Z1)")
    plt.title("Pareto Front Z1 vs Z2 (NSGA-II with Synergy Z3)")
    plt.grid()
    plt.show()
