from deap import base, creator, tools, algorithms
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Fungsi Objektif:
def evaluate(individual, project_data):
    n = len(project_data)
    Z1 = 0  # total fuzzy-adjusted ROI
    Z2 = 0  # total cost after synergy and risk
    for i in range(n):
        x_p = individual[i*5]         # binary selection
        α = individual[i*5 + 1]
        β = individual[i*5 + 2]
        γ = individual[i*5 + 3]
        start = int(individual[i*5 + 4])

        if x_p < 0.5:
            continue  # project not selected

        # Load project attributes
        proj = project_data[i]
        ROI = proj['roi']
        cost = proj['cost']
        synergy = proj['synergy']
        risk = proj['risk']

        # Z1: ROI (maximize, so negate for NSGA-II)
        Z1 += ROI * x_p

        # Z2: Cost with synergy adjustment + simple risk penalty
        adjusted_cost = cost - synergy
        risk_penalty = adjusted_cost * risk
        Z2 += (α + β + γ) * adjusted_cost + risk_penalty

    return -Z1, Z2  
# NSGA-II minimizes both

# Konfigurasi masalah
NUM_PROJECTS = 10

creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))  # Z1 (max) jadi -1
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

# Variabel: [x, α, β, γ, s] per proyek
def init_individual():
    ind = []
    for _ in range(NUM_PROJECTS):
        x = random.randint(0, 1)
        α = random.uniform(0, 1)
        β = random.uniform(0, 1 - α)
        γ = 1 - α - β
        s = random.randint(1, 7)  # asumsi 7 period
        ind += [x, α, β, γ, s]
    return ind

toolbox.register("individual", tools.initIterate, creator.Individual, init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Dummy data proyek
project_data = [{
    'roi': random.uniform(5, 15),
    'cost': random.uniform(100, 300),
    'synergy': random.uniform(5, 20),
    'risk': random.uniform(0.1, 0.3),
} for _ in range(NUM_PROJECTS)]

toolbox.register("evaluate", evaluate, project_data=project_data)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=1, indpb=0.1)
toolbox.register("select", tools.selNSGA2)

# Main loop
def main():
    pop = toolbox.population(n=100)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    pop, log = algorithms.eaMuPlusLambda(
        pop, toolbox, mu=100, lambda_=200, cxpb=0.7, mutpb=0.2,
        ngen=50, stats=stats, halloffame=hof, verbose=True
    )

    return pop, log, hof

if __name__ == "__main__":
    final_pop, logbook, pareto = main()

#Visualisasi Pareto Front
f1 = [-ind.fitness.values[0] for ind in pareto]  # reverse Z1
f2 = [ind.fitness.values[1] for ind in pareto]

plt.scatter(f2, f1, c="red")
plt.xlabel("Risk-Informed Cost (Z2)")
plt.ylabel("Expected Portfolio Value (Z1)")
plt.title("Pareto Front from NSGA-II for IFPOM")
plt.grid()
plt.show()

#debuging
z1s = [-ind.fitness.values[0] for ind in final_pop]
z2s = [ind.fitness.values[1] for ind in final_pop]

sns.scatterplot(x=z2s, y=z1s)
plt.title("Distribusi Seluruh Populasi Akhir")
plt.xlabel("Z2 (Risk-Informed Cost)")
plt.ylabel("Z1 (Expected Value)")
plt.grid()
plt.show()


