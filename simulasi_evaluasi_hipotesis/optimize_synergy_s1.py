import itertools
import pandas as pd
from copy import deepcopy
import numpy as np
from original_projects import load_project_data
from load_synergy_matrix import load_synergy_matrix
from common_ifpom import (
    initialize_ifpom, evaluate_individual, update_ideal_point, moead_generation
)
from set_h1_scenario import set_h1_scenario

def compute_risks(p, w_t=0.6, w_f=0.4):
    p['risk_tech'] = ((9 - p['trl']) / 8) * p['complexity']
    p['risk_fin'] = (
        p['alpha'] * 0.0 + p['beta'] * 0.3 + p['theta'] * 1.0 +
        p['gamma'] * 0.1 + p['delta'] * 0.6
    )
    p['risk'] = max(0.05, w_t * p['risk_tech'] + w_f * p['risk_fin'])

def run_synergy_simulation(config_id, scenario_code, synergy_weight, generations=100, pop_size=50):
    original_projects = load_project_data()
    delta_matrix = load_synergy_matrix()
    projects = deepcopy(original_projects)

    set_h1_scenario(scenario_code, projects)
    for p in projects:
        compute_risks(p)

    population, weights, neighbors = initialize_ifpom(pop_size=pop_size, num_projects=len(projects))
    ideal = [0.0, float('inf'), 0.0]

    for ind in population:
        ind['Z'] = evaluate_individual(ind, projects, delta_matrix, n_samples=10)
        update_ideal_point(ind['Z'], ideal)

    z_min = list(population[0]['Z'])
    z_max = list(population[0]['Z'])
    for ind in population:
        for k in range(3):
            z_min[k] = min(z_min[k], ind['Z'][k])
            z_max[k] = max(z_max[k], ind['Z'][k])

    for gen in range(generations):
        moead_generation(population, projects, delta_matrix, weights, neighbors, ideal, gen, generations, z_min, z_max)

    best = max(population, key=lambda ind: ind['Z'][0])
    print(f"[Config {config_id}] Scenario={scenario_code}, λ={synergy_weight} → Z1={best['Z'][0]:.2f}, Z2={best['Z'][1]:.2f}, Z3={best['Z'][2]:.2f}")
    return {
        'ConfigID': config_id,
        'Scenario': scenario_code,
        'Lambda': synergy_weight,
        'Z1': best['Z'][0],
        'Z2': best['Z'][1],
        'Z3': best['Z'][2]
    }

def run_synergy_experiments():
    scenarios = ["S1.1", "S1.2", "S1.3", "S1.4"]
    synergy_weights = [0.5, 1.0, 2.5, 5.0, 7.5]
    configs = list(itertools.product(scenarios, synergy_weights))

    results = []
    for i, (scenario_code, synergy_weight) in enumerate(configs):
        result = run_synergy_simulation(i+1, scenario_code, synergy_weight)
        results.append(result)

    df = pd.DataFrame(results)
    #df.to_csv("s1_synergy_tuning_results_2.csv", index=False)
    print("✅ All results saved to s1_synergy_tuning_results.csv")

    # === Optional Visualization ===
    import matplotlib.pyplot as plt
    pivot_z1 = df.pivot(index="Lambda", columns="Scenario", values="Z1").sort_index()
    pivot_z1.plot(kind='bar', figsize=(12, 6))
    plt.title("Z₁ – Strategic Value per Scenario and Synergy Weight λ")
    plt.xlabel("Synergy Weight (λ)")
    plt.ylabel("Strategic Value (Z₁)")
    plt.legend(title="Scenario")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_synergy_experiments()
