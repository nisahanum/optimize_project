import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters for 10 sample portfolio projects with Loan 1 Scheme
num_loan_sample_projects = 10  # Number of projects
loan_project_ids_sample = [f"L{str(i+1).zfill(3)}" for i in range(num_loan_sample_projects)]
loan_amounts_sample = np.random.uniform(100_000, 500_000, num_loan_sample_projects)
roi_values_loan_sample = np.random.uniform(0.13, 0.2, num_loan_sample_projects)
loan_term_sample = 3  # Loan term for Type 1 Loan
loan_interest_rate = 0.08  # Annual interest rate
grace_period = 1  # Grace period in years
early_repayment_threshold = 0.15  # Threshold ROI for early repayment

# Initialize adjusted cash flows
adjusted_cash_flows_loan_sample = []

for loan_amount, roi in zip(loan_amounts_sample, roi_values_loan_sample):
    annual_principal = loan_amount / loan_term_sample  # Equal principal repayment
    remaining_balance = loan_amount
    annual_cash_flows = []

    for year in range(1, loan_term_sample + 1):
        if year <= grace_period:
            # During grace period, no repayments are made
            interest_payment = remaining_balance * loan_interest_rate
            annual_cash_flow = round(interest_payment, 2)
        else:
            # Post grace period, principal + interest repayment
            interest_payment = remaining_balance * loan_interest_rate
            principal_payment = annual_principal
            total_payment = interest_payment + principal_payment
            annual_cash_flow = round(total_payment, 2)
            # Update remaining balance
            remaining_balance -= principal_payment

        # Early repayment logic
        if roi > early_repayment_threshold and remaining_balance > 0 and year > grace_period:
            remaining_balance = 0  # Pay off remaining balance early
            annual_cash_flow += remaining_balance * loan_interest_rate  # Adjust for one-time final payment

        annual_cash_flows.append(annual_cash_flow)

    adjusted_cash_flows_loan_sample.append(annual_cash_flows)

# Create an adjusted DataFrame for Loan 1 portfolio
adjusted_portfolio_loan_sample = pd.DataFrame({
    "Project_ID": loan_project_ids_sample,
    "Loan_Amount": loan_amounts_sample,
    "ROI": roi_values_loan_sample,
    **{f"Year_{i+1}": [adjusted_cash_flows_loan_sample[p][i] for p in range(num_loan_sample_projects)] for i in range(loan_term_sample)}
})

# Save adjusted data to CSV
adjusted_portfolio_loan_sample.to_csv("adjusted_loan1_sample_portfolio.csv", index=False)
print("Adjusted Loan 1 portfolio data has been saved to 'adjusted_loan1_sample_portfolio.csv'")

# Visualizing the adjusted cash flows
plt.figure(figsize=(10, 6))
for project_index in range(num_loan_sample_projects):
    plt.plot(
        range(1, loan_term_sample + 1),
        adjusted_cash_flows_loan_sample[project_index],
        label=f"Project {loan_project_ids_sample[project_index]}"
    )

plt.title("Adjusted Cash Flow for Loan 1 Investment Projects")
plt.xlabel("Year")
plt.ylabel("Cash Flow (USD)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.show()
