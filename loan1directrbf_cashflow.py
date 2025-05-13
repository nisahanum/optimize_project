import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters for Hybrid Financing
num_hybrid_projects = 10  # Number of projects
hybrid_project_ids = [f"HYB{str(i+1).zfill(3)}" for i in range(num_hybrid_projects)]

# Assign random weights for each financing type
weights = {
    "Loan_1": np.random.uniform(0.3, 0.5, num_hybrid_projects),  # Loan 1 contribution (30%-50%)
    "Direct_Investment": np.random.uniform(0.3, 0.5, num_hybrid_projects),  # Direct Investment (30%-50%)
    "RBF": np.random.uniform(0.1, 0.2, num_hybrid_projects),  # RBF contribution (10%-20%)
}

# Normalize weights to sum to 1 for each project
total_weights = weights["Loan_1"] + weights["Direct_Investment"] + weights["RBF"]
weights = {k: weights[k] / total_weights for k in weights}

# Simulated hybrid financing details
initial_investments_hybrid = np.random.uniform(500_000, 3_000_000, num_hybrid_projects)  # Total investment
roi_hybrid = np.random.uniform(0.15, 0.25, num_hybrid_projects)  # Hybrid ROI
expected_revenues_hybrid = np.random.uniform(300_000, 1_500_000, num_hybrid_projects)  # Expected revenues
loan_interest_rate = 0.08  # Annual interest rate for Loan 1
loan_term_sample = 3  # Loan term for 3 years
grace_period = 1  # Grace period for loan repayments

# Generate hybrid cash flows
hybrid_cash_flows = []
for i in range(num_hybrid_projects):
    loan_amount = initial_investments_hybrid[i] * weights["Loan_1"][i]
    direct_investment_amount = initial_investments_hybrid[i] * weights["Direct_Investment"][i]
    rbf_amount = initial_investments_hybrid[i] * weights["RBF"][i]
    rbf_revenue_share = np.random.uniform(0.05, 0.15)  # Revenue share for RBF

    # Loan repayments
    loan_principal = loan_amount / loan_term_sample
    loan_balance = loan_amount
    loan_cash_flows = []

    for year in range(loan_term_sample):
        if year < grace_period:
            # Grace period: only interest payments
            interest_payment = loan_balance * loan_interest_rate
            loan_cash_flows.append(interest_payment)
        else:
            # Regular repayments after grace period
            interest_payment = loan_balance * loan_interest_rate
            total_payment = loan_principal + interest_payment
            loan_balance -= loan_principal
            loan_cash_flows.append(total_payment)

    # Direct Investment returns
    direct_investment_cash_flows = [direct_investment_amount * roi_hybrid[i] for _ in range(loan_term_sample)]

    # RBF repayments
    rbf_cash_flows = [expected_revenues_hybrid[i] * rbf_revenue_share for _ in range(loan_term_sample)]

    # Combine cash flows
    hybrid_cash_flows.append([
        round(loan_cash_flows[year] + direct_investment_cash_flows[year] + rbf_cash_flows[year], 2)
        for year in range(loan_term_sample)
    ])

# Create a DataFrame for Hybrid Financing projects
hybrid_portfolio = pd.DataFrame({
    "Project_ID": hybrid_project_ids,
    "Total_Investment": initial_investments_hybrid,
    "Loan_1_Share": weights["Loan_1"],
    "Direct_Investment_Share": weights["Direct_Investment"],
    "RBF_Share": weights["RBF"],
    **{f"Year_{i+1}": [hybrid_cash_flows[p][i] for p in range(num_hybrid_projects)] for i in range(loan_term_sample)}
})

# Save Hybrid Financing data to CSV
hybrid_portfolio.to_csv("hybrid_financing_portfolio.csv", index=False)
print("Hybrid Financing portfolio data has been saved to 'hybrid_financing_portfolio.csv'")

# Visualizing the cash flows for Hybrid Financing projects
plt.figure(figsize=(10, 6))
for project_index in range(num_hybrid_projects):
    plt.plot(
        range(1, loan_term_sample + 1),
        hybrid_cash_flows[project_index],
        label=f"Project {hybrid_project_ids[project_index]}"
    )

plt.title("Cash Flow for Hybrid Financing Projects")
plt.xlabel("Year")
plt.ylabel("Cash Flow (USD)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.show()
