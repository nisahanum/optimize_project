import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For visualization

import matplotlib.pyplot as plt

# Parameters for 10 sample portfolio projects
num_sample_projects = 10  # Number of projects
sample_project_ids = [f"P{str(i+1).zfill(3)}" for i in range(num_sample_projects)]
initial_investments_sample = np.random.uniform(1_000_000, 5_000_000, num_sample_projects)
roi_values_sample = np.random.uniform(0.15, 0.30, num_sample_projects)
years_sample = 5  # Cash flow projection for 5 years

# Generate cash flows for sample projects
cash_flows_sample = [
    [round(initial_investment * roi, 2) for _ in range(years_sample)]
    for initial_investment, roi in zip(initial_investments_sample, roi_values_sample)
]

# Create a DataFrame for sample portfolio
portfolio_sample = pd.DataFrame({
    "Project_ID": sample_project_ids,
    "Initial_Investment": initial_investments_sample,
    "ROI": roi_values_sample,
    **{f"Year_{i+1}": [cash_flows_sample[p][i] for p in range(num_sample_projects)] for i in range(years_sample)}
})

# Visualizing the cash flow for the sample projects
for project_index in range(num_sample_projects):
    plt.plot(
        range(1, years_sample + 1),
        cash_flows_sample[project_index],
        label=f"Project {sample_project_ids[project_index]}"
    )

plt.title("Cash Flow for Direct Investment Projects")
plt.xlabel("Year")
plt.ylabel("Cash Flow (USD)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

print(portfolio_sample)  # Print the DataFrame in the terminal
portfolio_sample.to_csv("direct_investment_sample_portfolio.csv", index=False)
print("Sample portfolio data has been saved to 'direct_investment_sample_portfolio.csv'")
