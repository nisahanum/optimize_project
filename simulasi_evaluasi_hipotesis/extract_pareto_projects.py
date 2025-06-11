import json
import pandas as pd
from original_projects import load_project_data

# === Fungsi untuk load populasi hasil MOEA/D ===
def load_population(filepath="population_final.json"):
    with open(filepath, 'r') as f:
        return json.load(f)

# === Fungsi untuk ekstrak solusi dan proyek terpilih ===
def extract_selected_projects(population, projects):
    project_names = [f"{p['id']} - {p['benefit_group']}" for p in projects]
    result = []
    for i, ind in enumerate(population):
        selected = [project_names[j] for j, val in enumerate(ind['x']) if val == 1]
        result.append({
            'Solution': f'Individual {i+1}',
            'Z1 (Strategic Value)': ind['Z'][0],
            'Z2 (Financial Cost)': ind['Z'][1],
            'Z3 (Synergy)': ind['Z'][2],
            'Selected Projects': "; ".join(selected)
        })
    return pd.DataFrame(result)

# === Main Execution ===
if __name__ == "__main__":
    population = load_population("population_final.json")
    projects = load_project_data()
    df = extract_selected_projects(population, projects)

    # Simpan sebagai CSV
    df.to_csv("selected_projects_summary.csv", index=False)
    print("✅ File saved: selected_projects_summary.csv")

    # Preview di terminal (optional)
    print(df.head())
