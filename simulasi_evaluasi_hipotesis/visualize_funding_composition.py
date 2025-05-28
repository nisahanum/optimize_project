import pandas as pd
import matplotlib.pyplot as plt
from original_projects import load_project_data
from set_h2_scenario import set_h2_scenario

def compute_funding_composition():
    scenarios = ["S2.1", "S2.2", "S2.3", "S2.4", "S2.5", "S2.6"]
    records = []

    for code in scenarios:
        projects = load_project_data()
        set_h2_scenario(code, projects)
        
        avg_alpha = sum(p['alpha'] for p in projects) / len(projects)
        avg_beta = sum(p['beta'] for p in projects) / len(projects)
        avg_theta = sum(p['theta'] for p in projects) / len(projects)
        avg_gamma = sum(p['gamma'] for p in projects) / len(projects)
        avg_delta = sum(p['delta'] for p in projects) / len(projects)

        records.append({
            'Scenario': code,
            'Alpha (Equity)': avg_alpha,
            'Beta (Soft Loan)': avg_beta,
            'Theta (Vendor)': avg_theta,
            'Gamma (Grant)': avg_gamma,
            'Delta (PPP)': avg_delta,
        })

    return pd.DataFrame(records)

def plot_funding_composition(df):
    df.set_index('Scenario').plot(kind='bar', figsize=(10, 6))
    plt.ylabel("Average Funding Proportion")
    plt.title("Funding Composition Across Scenarios (S2.1–S2.6)")
    plt.xticks(rotation=0)
    plt.ylim(0, 1.0)
    plt.legend(title="Funding Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df_funding = compute_funding_composition()
    print(df_funding.round(3))  # Optional: Show data in terminal
    plot_funding_composition(df_funding)
