
# run_h1_simulation.py

from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# === Simulasi Hipotesis 1 ===
def run_all_h1_scenarios():
    h1_scenarios = ["S1.1", "S1.2", "S1.3", "S1.4"]
    results = []

    original_projects = load_project_data()
    delta_matrix = load_synergy_matrix(num_projects=len(original_projects))

    for sc in h1_scenarios:
        print(f"\n=== Running Scenario {sc} ===")
        projects = deepcopy(original_projects)
        set_h1_scenario(sc, projects)

        for p in projects:
            compute_risks(p)

        population, weights, neighbors = initialize_ifpom(
            pop_size=50, num_projects=len(projects)
        )
        ideal = [0.0, float('inf'), 0.0]

        for ind in population:
            ind['Z'] = evaluate_individual(ind, projects, delta_matrix)
            update_ideal_point(ind['Z'], ideal)

        for gen in range(100):
            moead_generation(population, projects, delta_matrix, weights, neighbors, ideal)

        best = max(population, key=lambda ind: ind['Z'][0])
        print(f"{sc} → Z1 = {best['Z'][0]:.2f}, Z2 = {best['Z'][1]:.2f}, Z3 = {best['Z'][2]:.2f}")
        results.append({
            'scenario': sc,
            'Z1': best['Z'][0],
            'Z2': best['Z'][1],
            'Z3': best['Z'][2]
        })

    return results

# === Visualisasi Hasil H1 ===
def plot_results(results):
    df = pd.DataFrame(results)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    ax[0].bar(df['scenario'], df['Z1'], color='skyblue')
    ax[0].set_title('Z1 - Strategic Value')
    ax[0].set_ylabel('Score')
    ax[0].set_ylim(0, df['Z1'].max() + 100)

    ax[1].bar(df['scenario'], df['Z2'], color='salmon')
    ax[1].set_title('Z2 - Risk-Adjusted Financial Cost')
    ax[1].set_ylabel('Cost')
    ax[1].set_ylim(0, df['Z2'].max() + 100)

    ax[2].bar(df['scenario'], df['Z3'], color='lightgreen')
    ax[2].set_title('Z3 - Total Synergy')
    ax[2].set_ylabel('Synergy')
    ax[2].set_ylim(0, df['Z3'].max() + 100)

    plt.suptitle("Hasil Simulasi Hipotesis H1 per Skenario", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# === Main Program ===
if __name__ == "__main__":
    results = run_all_h1_scenarios()
    plot_results(results)
