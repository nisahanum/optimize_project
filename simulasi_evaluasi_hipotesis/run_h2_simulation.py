# run_h2_simulation.py

from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from original_projects import load_project_data
from load_synergy_matrix import load_synergy_matrix
from common_ifpom import (
    initialize_ifpom, evaluate_individual, update_ideal_point,
    moead_generation
)

# === Skenario H2: Definisi dan Setup ===
def set_h2_scenario(scenario_code, projects):
    for p in projects:
        if scenario_code == "S2.1":
            p['alpha'], p['beta'], p['theta'], p['gamma'], p['delta'] = 0.3, 0.3, 0.4, 0.0, 0.0
        elif scenario_code == "S2.2":
            p['alpha'], p['beta'], p['theta'], p['gamma'], p['delta'] = 0.0, 1.0, 0.0, 0.0, 0.0
        elif scenario_code == "S2.3":
            p['alpha'], p['beta'], p['theta'], p['gamma'], p['delta'] = 0.0, 0.0, 1.0, 0.0, 0.0
        elif scenario_code == "S2.4":
            mix = np.random.dirichlet([1, 1, 1, 1, 1])
            mix[2] = min(mix[2], 0.4)
            total = sum(mix)
            mix = [m / total for m in mix]
            p['alpha'], p['beta'], p['theta'], p['gamma'], p['delta'] = mix

def compute_risks(p, w_t=0.6, w_f=0.4):
    p['risk_tech'] = ((9 - p['trl']) / 8) * p['complexity']
    p['risk_fin'] = (
        p['alpha'] * 0.0 + p['beta'] * 0.3 + p['theta'] * 1.0 +
        p['gamma'] * 0.1 + p['delta'] * 0.6
    )
    p['risk'] = max(0.05, w_t * p['risk_tech'] + w_f * p['risk_fin'])

# === Eksekusi Semua Skenario H2 ===
def run_all_h2_scenarios():
    h2_scenarios = ["S2.1", "S2.2", "S2.3", "S2.4"]
    results = []

    original_projects = load_project_data()
    delta_matrix = load_synergy_matrix(num_projects=len(original_projects))

    for sc in h2_scenarios:
        print(f"\n=== Running Scenario {sc} ===")
        projects = deepcopy(original_projects)
        set_h2_scenario(sc, projects)

        for p in projects:
            compute_risks(p)

        population, weights, neighbors = initialize_ifpom(
            pop_size=50,
            num_projects=len(projects)
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

# === MAIN PROGRAM ===
if __name__ == "__main__":
    results = run_all_h2_scenarios()

    # === OPSIONAL: Simpan ke file CSV ===
    

    # === Visualisasi ===
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

    plt.suptitle("Hasil Simulasi Hipotesis H2 per Skenario", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    
#simpan ke file csv
#df_save = pd.DataFrame(results)
#df_save.to_csv("hasil_simulasi_H2_ver2.csv", index=False)
#print("✅ Hasil simulasi H2 disimpan ke: hasil_simulasi_H2.csv")