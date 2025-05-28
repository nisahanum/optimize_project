import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from copy import deepcopy

from original_projects import load_project_data
from set_h1_scenario import set_h1_scenario
from load_synergy_matrix import load_synergy_matrix
from common_ifpom import (
    initialize_ifpom,
    evaluate_individual,
    update_ideal_point,
    moead_generation
)

# === Injected Risk Computation Logic ===
def compute_risks(p, w_t=0.6, w_f=0.4):
    p['risk_tech'] = ((9 - p['trl']) / 8) * p['complexity']
    p['risk_fin'] = (
        p['alpha'] * 0.0 + p['beta'] * 0.3 + p['theta'] * 1.0 +
        p['gamma'] * 0.1 + p['delta'] * 0.6
    )
    p['risk'] = max(0.05, w_t * p['risk_tech'] + w_f * p['risk_fin'])

# === MOEA/D Runner with Logging ===
def run_moead_with_logging(scenario_code="S1.4", generations=100, pop_size=50):
    original_projects = load_project_data()
    projects = deepcopy(original_projects)

    # Scenario-specific synergy setup
    set_h1_scenario(scenario_code, projects)

    # Risk assignment
    for p in projects:
        compute_risks(p)

    delta_matrix = load_synergy_matrix()
    population, weights, neighbors = initialize_ifpom(pop_size, num_projects=len(projects))
    ideal = [0.0, float('inf'), 0.0]

    # Evaluate initial population
    for ind in population:
        ind['Z'] = evaluate_individual(ind, projects, delta_matrix)
        update_ideal_point(ind['Z'], ideal)

    z_min = list(population[0]['Z'])
    z_max = list(population[0]['Z'])
    for ind in population:
        for k in range(3):
            z_min[k] = min(z_min[k], ind['Z'][k])
            z_max[k] = max(z_max[k], ind['Z'][k])

    logbook = []
    for gen in range(1, generations + 1):
        moead_generation(population, projects, delta_matrix, weights, neighbors, ideal, gen, generations, z_min, z_max)
        if gen in [10, 20, 30, 50, 70, 80, 90, 100]:
            for ind in population:
                logbook.append({
                    'Generation': gen,
                    'Z1': ind['Z'][0],
                    'Z2': ind['Z'][1],
                    'Z3': ind['Z'][2]
                })

    return logbook


# === Visualizer ===
def plot_logbook(logbook):
    df = pd.DataFrame(logbook)
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Z1 / Z3', color='tab:blue')
    ax1.plot(df['Generation'], df['Z1'], label='Z1 Strategic Value', marker='o')
    ax1.plot(df['Generation'], df['Z3'], label='Z3 Synergy', marker='x')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Z2 Cost', color='tab:red')
    ax2.plot(df['Generation'], df['Z2'], label='Z2 Cost', color='tab:red', linestyle='--', marker='s')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    plt.title("Pareto Progression on Strategic Value, Cost, and Synergy")
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    plt.grid(True)
    plt.show()

# === Execute ===
if __name__ == "__main__":
    logbook = run_moead_with_logging(scenario_code="S1.4", generations=100, pop_size=50)

    # Save logbook to CSV before visualization
    df_logbook = pd.DataFrame(logbook)
    df_logbook.to_csv("moead_logbook_output.csv", index=False)

    plot_logbook(logbook)
