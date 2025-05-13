import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Parameters for simulation
num_projects = 10
project_ids = [f"P{str(i+1).zfill(3)}" for i in range(num_projects)]

# Simulated values for projects
initial_investments = np.random.uniform(1_000_000, 5_000_000, num_projects)
roi_direct_investment = np.random.uniform(0.15, 0.30, num_projects)  # ROI for direct investment
roi_hybrid = np.random.uniform(0.13, 0.25, num_projects)  # ROI for hybrid financing

# Hybrid financing weights
weights_hybrid = {
    "Loan_1": np.random.uniform(0.3, 0.5, num_projects),
    "Direct_Investment": np.random.uniform(0.3, 0.5, num_projects),
    "RBF": np.random.uniform(0.1, 0.2, num_projects),
}

# Normalize weights to sum to 1
total_weights = weights_hybrid["Loan_1"] + weights_hybrid["Direct_Investment"] + weights_hybrid["RBF"]
weights_hybrid = {k: weights_hybrid[k] / total_weights for k in weights_hybrid}

# Simulate cash flows for Direct Investment
cash_flows_direct_investment = [
    [round(initial_investments[i] * roi_direct_investment[i], 2) for _ in range(5)] for i in range(num_projects)
]

# Simulate cash flows for Hybrid Financing
cash_flows_hybrid = []
for i in range(num_projects):
    loan_amount = initial_investments[i] * weights_hybrid["Loan_1"][i]
    direct_investment_amount = initial_investments[i] * weights_hybrid["Direct_Investment"][i]
    rbf_amount = initial_investments[i] * weights_hybrid["RBF"][i]
    rbf_revenue_share = np.random.uniform(0.05, 0.15)

    loan_principal = loan_amount / 5
    loan_balance = loan_amount
    loan_cash_flows = []

    for year in range(5):
        interest_payment = loan_balance * 0.08
        total_payment = loan_principal + interest_payment
        loan_balance -= loan_principal
        loan_cash_flows.append(total_payment)

    direct_investment_cash_flows = [direct_investment_amount * roi_hybrid[i] for _ in range(5)]
    rbf_cash_flows = [rbf_amount * rbf_revenue_share for _ in range(5)]

    hybrid_cash_flows = [
        round(loan_cash_flows[year] + direct_investment_cash_flows[year] + rbf_cash_flows[year], 2)
        for year in range(5)
    ]
    cash_flows_hybrid.append(hybrid_cash_flows)

# Calculate metrics: ROI, NPV, Payback Period
results = []
for i in range(num_projects):
    # Direct Investment Metrics
    total_direct_return = sum(cash_flows_direct_investment[i])
    roi_direct = total_direct_return / initial_investments[i]
    npv_direct = total_direct_return - initial_investments[i]
    payback_direct = initial_investments[i] / (cash_flows_direct_investment[i][0])

    # Hybrid Financing Metrics
    total_hybrid_return = sum(cash_flows_hybrid[i])
    roi_hybrid_calc = total_hybrid_return / initial_investments[i]
    npv_hybrid = total_hybrid_return - initial_investments[i]
    payback_hybrid = initial_investments[i] / (cash_flows_hybrid[i][0])

    results.append({
        "Project_ID": project_ids[i],
        "Initial_Investment": initial_investments[i],
        "ROI_Direct": roi_direct,
        "NPV_Direct": npv_direct,
        "Payback_Direct": payback_direct,
        "ROI_Hybrid": roi_hybrid_calc,
        "NPV_Hybrid": npv_hybrid,
        "Payback_Hybrid": payback_hybrid,
    })

# Convert results to DataFrame for analysis
comparison_df = pd.DataFrame(results)


# Print the results in the terminal
print(comparison_df)

# Save the results to a CSV file
comparison_df.to_csv("direct_vs_hybrid_comparison.csv", index=False)
print("Comparison data saved to 'direct_vs_hybrid_comparison.csv'")

# Visualizing ROI, NPV, and Payback Period from the earlier simulation results

# ROI Comparison
plt.figure(figsize=(10, 6))
plt.bar(comparison_df["Project_ID"], comparison_df["ROI_Direct"] * 100, alpha=0.7, label="Direct Investment ROI", width=0.4, align='center')
plt.bar(comparison_df["Project_ID"], comparison_df["ROI_Hybrid"] * 100, alpha=0.7, label="Hybrid Financing ROI", width=0.4, align='edge')
plt.title("ROI Comparison: Direct Investment vs Hybrid Financing")
plt.ylabel("ROI (%)")
plt.xlabel("Projects")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# NPV Comparison
plt.figure(figsize=(10, 6))
plt.bar(comparison_df["Project_ID"], comparison_df["NPV_Direct"], alpha=0.7, label="Direct Investment NPV", width=0.4, align='center')
plt.bar(comparison_df["Project_ID"], comparison_df["NPV_Hybrid"], alpha=0.7, label="Hybrid Financing NPV", width=0.4, align='edge')
plt.title("NPV Comparison: Direct Investment vs Hybrid Financing")
plt.ylabel("NPV (USD)")
plt.xlabel("Projects")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Payback Period Comparison
plt.figure(figsize=(10, 6))
plt.bar(comparison_df["Project_ID"], comparison_df["Payback_Direct"], alpha=0.7, label="Direct Investment Payback Period", width=0.4, align='center')
plt.bar(comparison_df["Project_ID"], comparison_df["Payback_Hybrid"], alpha=0.7, label="Hybrid Financing Payback Period", width=0.4, align='edge')
plt.title("Payback Period Comparison: Direct Investment vs Hybrid Financing")
plt.ylabel("Payback Period (Years)")
plt.xlabel("Projects")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
