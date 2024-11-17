import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simulate data for hybrid projects (if not already available)
num_hybrid_projects = 10  # Number of projects
loan_term_sample = 3  # Loan term
weights = {
    "Loan_1": np.random.uniform(0.3, 0.5, num_hybrid_projects),
    "Direct_Investment": np.random.uniform(0.3, 0.5, num_hybrid_projects),
    "RBF": np.random.uniform(0.1, 0.2, num_hybrid_projects),
}
total_weights = weights["Loan_1"] + weights["Direct_Investment"] + weights["RBF"]
weights = {k: weights[k] / total_weights for k in weights}

initial_investments_hybrid = np.random.uniform(500_000, 3_000_000, num_hybrid_projects)
roi_hybrid = np.random.uniform(0.15, 0.25, num_hybrid_projects)
expected_revenues_hybrid = np.random.uniform(300_000, 1_500_000, num_hybrid_projects)

# Generate cash flows for hybrid projects
hybrid_cash_flows = []
for i in range(num_hybrid_projects):
    loan_amount = initial_investments_hybrid[i] * weights["Loan_1"][i]
    direct_investment_amount = initial_investments_hybrid[i] * weights["Direct_Investment"][i]
    rbf_amount = initial_investments_hybrid[i] * weights["RBF"][i]
    rbf_revenue_share = np.random.uniform(0.05, 0.15)

    loan_principal = loan_amount / loan_term_sample
    loan_balance = loan_amount
    loan_cash_flows = []
    for year in range(loan_term_sample):
        interest_payment = loan_balance * 0.08
        total_payment = loan_principal + interest_payment if year >= 1 else interest_payment
        loan_balance -= loan_principal
        loan_cash_flows.append(total_payment)

    direct_investment_cash_flows = [direct_investment_amount * roi_hybrid[i] for _ in range(loan_term_sample)]
    rbf_cash_flows = [expected_revenues_hybrid[i] * rbf_revenue_share for _ in range(loan_term_sample)]

    hybrid_cash_flows.append([
        round(loan_cash_flows[year] + direct_investment_cash_flows[year] + rbf_cash_flows[year], 2)
        for year in range(loan_term_sample)
    ])

# Summarize cash flows for visualization
loan_cash_flows_summary = [sum([hybrid_cash_flows[p][year] * weights["Loan_1"][p] for year in range(loan_term_sample)]) for p in range(num_hybrid_projects)]
direct_investment_cash_flows_summary = [sum([hybrid_cash_flows[p][year] * weights["Direct_Investment"][p] for year in range(loan_term_sample)]) for p in range(num_hybrid_projects)]
rbf_cash_flows_summary = [sum([hybrid_cash_flows[p][year] * weights["RBF"][p] for year in range(loan_term_sample)]) for p in range(num_hybrid_projects)]
hybrid_total_cash_flows = [sum(hybrid_cash_flows[p]) for p in range(num_hybrid_projects)]

# Visualization
x_labels = [f"Project {i+1}" for i in range(num_hybrid_projects)]
x_positions = np.arange(len(x_labels))

plt.figure(figsize=(12, 8))

# Plot each financing type
plt.bar(x_positions - 0.2, loan_cash_flows_summary, width=0.2, label="Loan 1", align='center')
plt.bar(x_positions, direct_investment_cash_flows_summary, width=0.2, label="Direct Investment", align='center')
plt.bar(x_positions + 0.2, rbf_cash_flows_summary, width=0.2, label="RBF", align='center')
plt.plot(x_positions, hybrid_total_cash_flows, label="Hybrid Total", marker='o', color='black', linestyle='--', linewidth=1)

# Add labels and legend
plt.title("Cash Flow Comparisons: Loan 1, Direct Investment, RBF, and Hybrid")
plt.xlabel("Projects")
plt.ylabel("Total Cash Flow (USD)")
plt.xticks(x_positions, x_labels, rotation=45)
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()
