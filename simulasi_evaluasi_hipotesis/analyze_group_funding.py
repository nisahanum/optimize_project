import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from original_projects import load_project_data
from load_synergy_matrix import load_synergy_matrix
from common_ifpom import evaluate_individual
from risk_utils import compute_risks

# === Funding Mix Logic ===
def define_funding_mix(group, variant):
    if variant == 'alpha':
        return 1.0, 0.0, 0.0, 0.0, 0.0
    elif variant == 'beta':
        return 0.0, 1.0, 0.0, 0.0, 0.0
    elif variant == 'theta':
        return 0.0, 0.0, 1.0, 0.0, 0.0
    elif variant == 'gamma':
        return 0.0, 0.0, 0.0, 1.0, 0.0
    elif variant == 'delta':
        return 0.0, 0.0, 0.0, 0.0, 1.0
    elif variant == 'balanced':
        if group == "Operational Efficiency":
            return 0.2, 0.2, 0.2, 0.0, 0.4
        elif group == "Business Culture":
            return 0.3, 0.2, 0.2, 0.3, 0.0
        else:  # Customer Experience
            return 0.4, 0.3, 0.3, 0.0, 0.0
    else:
        raise ValueError(f"Unknown funding variant: {variant}")

# === Evaluation Loop ===
def analyze_group_funding():
    projects = load_project_data()
    delta_matrix = load_synergy_matrix()
    n_samples = 10

    variants = ['alpha', 'beta', 'theta', 'gamma', 'delta', 'balanced']
    benefit_groups = ['Operational Efficiency', 'Customer Experience', 'Business Culture']
    results = []

    variant_labels = {
        'alpha': 'Internal Equity',
        'beta': 'Soft Loan',
        'theta': 'Vendor Financing',
        'gamma': 'Grant',
        'delta': 'PPP',
        'balanced': 'Balanced Mix'
    }

    for group in benefit_groups:
        group_projects = [p for p in deepcopy(projects) if p['benefit_group'] == group]
        for variant in variants:
            for p in group_projects:
                alpha, beta, theta, gamma, delta = define_funding_mix(group, variant)

                # Original keys for model logic
                p['alpha'] = alpha
                p['beta'] = beta
                p['theta'] = theta
                p['gamma'] = gamma
                p['delta'] = delta

                # Optional: human-readable funding types (not used in model)
                p['Internal Equity'] = alpha
                p['Soft Loan'] = beta
                p['Vendor Financing'] = theta
                p['Grant'] = gamma
                p['PPP'] = delta

                compute_risks(p)

            dummy_ind = {'x': [1] * len(group_projects)}
            Z = evaluate_individual(dummy_ind, group_projects, delta_matrix, n_samples=n_samples)

            results.append({
                'BenefitGroup': group,
                'FundingVariant': variant_labels[variant],
                'Z1': Z[0],
                'Z2': Z[1],
                'Z3': Z[2]
            })

    return pd.DataFrame(results)

# === Main Execution ===
if __name__ == "__main__":
    df = analyze_group_funding()

    # Optional CSV export
    # df.to_csv("group_funding_analysis.csv", index=False)
    # print("✅ Results saved to group_funding_analysis.csv")

    # Plot: Separate axes for each Z objective
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    objectives = ['Z1', 'Z2', 'Z3']
    colors = ['steelblue', 'darkorange', 'seagreen']

    for i, obj in enumerate(objectives):

        sns.barplot(
            data=df,
            x="FundingVariant",
            y=obj,
            hue="BenefitGroup",
            ax=axes[i],
            palette="tab10"
        )
        axes[i].set_title(f"{obj} by Funding Type in Benefit Groups", fontsize=12)
        axes[i].set_ylabel(f"{obj} Value", fontsize=12)
        axes[i].tick_params(axis='both', labelsize=10)
        

        # Show legend only in middle (Z2) subplot
    for i in [0, 2]:
        legend = axes[i].get_legend()
        if legend:
            legend.remove()


    # Add legend only for Z2
    axes[1].legend(title="Benefit Group", fontsize=12, title_fontsize=13, loc='upper left')

# X-axis config
    axes[-1].set_xlabel("Funding Variant", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()