import itertools
import pandas as pd
from copy import deepcopy
import numpy as np
from original_projects import load_project_data
from load_synergy_matrix import load_synergy_matrix
from common_ifpom import (
    initialize_ifpom, evaluate_individual, update_ideal_point, moead_generation
)

def apply_adaptive_funding(projects, theta_cap, synergy_weight):
    for p in projects:
        risk = p.get('risk', 0.5)
        svs = p.get('svs', 60)
        fuzzy = p.get('fuzzy_cost', (2.0, 2.5, 3.0))
        cost_est = (fuzzy[0] + 2 * fuzzy[1] + fuzzy[2]) / 4

        if svs > 85 and risk < 0.4:
            alpha, beta, theta = 0.2, 0.2, 0.6
        elif risk > 0.6:
            alpha, beta, theta = 0.6, 0.3, 0.1
        else:
            alpha, beta, theta = 0.4, 0.3, 0.3

        if theta > theta_cap:
            excess = theta - theta_cap
            theta = theta_cap
            alpha += excess * 0.6
            beta += excess * 0.4

        total = alpha + beta + theta
        p['alpha'] = alpha / total
        p['beta'] = beta / total
        p['theta'] = theta / total
        p['gamma'] = 0.0
        p['delta'] = 0.0

def compute_risks(p, w_t=0.6, w_f=0.4):
    p['risk_tech'] = ((9 - p['trl']) / 8) * p['complexity']
    p['risk_fin'] = (
        p['alpha'] * 0.0 + p['beta'] * 0.3 + p['theta'] * 1.0 +
        p['gamma'] * 0.1 + p['delta'] * 0.6
    )
    p['risk'] = max(0.05, w_t * p['risk_tech'] + w_f * p['risk_fin'])

def run_simulation(config_id, theta_cap, synergy_weight, generations=100, pop_size=50):
    original_projects = load_project_data()
    delta_matrix = load_synergy_matrix()
    projects = deepcopy(original_projects)

    apply_adaptive_funding(projects, theta_cap, synergy_weight)
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
    print(f"[Config {config_id}] θ≤{theta_cap}, λ={synergy_weight} → Z1={best['Z'][0]:.2f}, Z2={best['Z'][1]:.2f}, Z3={best['Z'][2]:.2f}")
    return {
        'ConfigID': config_id,
        'ThetaCap': theta_cap,
        'SynergyWeight': synergy_weight,
        'Z1': best['Z'][0],
        'Z2': best['Z'][1],
        'Z3': best['Z'][2]
    }

def run_experiments():
    theta_caps = [0.4, 0.5, 0.6]
    synergy_weights = [1.0, 2.5, 5.0]
    configs = list(itertools.product(theta_caps, synergy_weights))

    results = []
    for i, (theta_cap, synergy_weight) in enumerate(configs):
        result = run_simulation(i+1, theta_cap, synergy_weight)
        results.append(result)

    df = pd.DataFrame(results)
    #df.to_csv("s2_4_monte_carlo_results_2.csv", index=False)
    print("✅ All results saved to s2_4_monte_carlo_results.csv")

if __name__ == "__main__":
    run_experiments()
