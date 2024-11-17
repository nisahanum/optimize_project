import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters for Revenue-Based Financing (RBF) simulation
num_rbf_projects = 10  # Number of projects
rbf_project_ids = [f"RBF{str(i+1).zfill(3)}" for i in range(num_rbf_projects)]
revenue_share = np.random.uniform(0.05, 0.15, num_rbf_projects)  # Revenue share percentage (5% to 15%)
expected_revenues = np.random.uniform(200_000, 2_000_000, num_rbf_projects)  # Expected annual revenues
loan_term_sample = 3  # Loan term (3 years)

# Generate cash flows for RBF projects
rbf_cash_flows = [
    [round(expected_revenue * share, 2) for _ in range(loan_term_sample)]
    for expected_revenue, share in zip(expected_revenues, revenue_share)
]

# Create a DataFrame for RBF projects
rbf_portfolio = pd.DataFrame({
    "Project_ID": rbf_project_ids,
    "Expected_Annual_Revenue": expected_revenues,
    "Revenue_Share": revenue_share,
    **{f"Year_{i+1}": [rbf_cash_flows[p][i] for p in range(num_rbf_projects)] for i in range(loan_term_sample)}
})

# Save RBF data to CSV
rbf_portfolio.to_csv("revenue_based_financing_portfolio.csv", index=False)
print("Revenue-Based Financing portfolio data has been saved to 'revenue_based_financing_portfolio.csv'")

# Visualizing the cash flows for RBF projects
plt.figure(figsize=(10, 6))
for project_index in range(num_rbf_projects):
    plt.plot(
        range(1, loan_term_sample + 1),
        rbf_cash_flows[project_index],
        label=f"Project {rbf_project_ids[project_index]}"
    )

plt.title("Cash Flow for Revenue-Based Financing Projects")
plt.xlabel("Year")
plt.ylabel("Cash Flow (USD)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.show()
