import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# ----------------------- Konfigurasi Masalah -----------------------

NUM_PROJECTS = 20

# Buat struktur fitness dan individual
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))  # -Z1 (maximize), Z2 (minimize)
creator.create("Individual", list, fitness=creator.FitnessMulti)
toolbox = base.Toolbox()

# Data proyek dengan variabilitas tinggi
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

# -------------------- Fungsi Evaluasi Bersih ------------------------

def evaluate(individual):
    total_roi = 0
    total_risk_adjusted_cost = 0

    for i in range(NUM_PROJECTS):
        x = int(round(individual[i*5]))
        a = individual[i*5 + 1]
        b = individual[i*5 + 2]
        c = individual[i*5 + 3]
        s = individual[i*5 + 4]

        if x == 0:
            continue

        proj = project_data[i]
        ROI = proj['roi']
        cost = proj['cost']
        risk = proj['risk']

        # Z1: Adjusted ROI (maximize)
        total_roi += ROI * (1 - risk)

        # Z2: Risk-adjusted cost (minimize)
        total_risk_adjusted_cost += (a + b + c) * cost * (1 + risk)

    return -total_roi, total_risk_adjusted_cost

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

    plt.scatter(f2, f1, c="red")
    plt.xlabel("Risk-Adjusted Cost (Z2)")
    plt.ylabel("Adjusted ROI (Z1)")
    plt.title("Clean Multi-Objective Pareto Front (NSGA-II IFPOM)")
    plt.grid()
    plt.show()
