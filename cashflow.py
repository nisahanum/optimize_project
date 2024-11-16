import pandas as pd
import matplotlib.pyplot as plt

# Define simulation parameters
years = [1, 2, 3, 4, 5]

# Revenue growth assumptions
revenues = [0, 50, 70, 90, 110]  # Revenue in millions starting Year 2
operational_cost_percentage = 0.30  # 30% of revenue as operational cost

# Financing types and proportions
equity_percentage = 0.4  # 40% financed by equity
debt_percentage = 0.3  # 30% financed by debt
rbf_percentage = 0.3  # 30% financed by RBF

# Debt repayment terms (5% interest, $100M principal repayment)
debt_annual_interest_rate = 0.05
debt_total_investment = 300 * debt_percentage  # Debt investment proportion

# RBF repayment rates (variable by year)
rbf_repayment_rates = [0.0, 0.10, 0.12, 0.15, 0.18]  # Increasing repayment percentages

# Initialize list for cash flow data
complex_cashflow = []

# Simulate for each year
for year in years:
    revenue = revenues[year - 1]
    operational_costs = revenue * operational_cost_percentage

    # Equity financing cash flow: no repayments
    equity_cash_flow = equity_percentage * revenue - operational_costs

    # Debt financing cash flow: includes fixed principal and interest repayments
    debt_interest_payment = debt_total_investment * debt_annual_interest_rate if year > 1 else 0
    debt_principal_repayment = debt_total_investment / 5 if year > 1 else 0
    debt_cash_flow = (
        (debt_percentage * revenue)
        - operational_costs
        - debt_interest_payment
        - debt_principal_repayment
    )

    # RBF cash flow: variable repayments based on revenue
    rbf_repayment = revenue * rbf_repayment_rates[year - 1] if year > 1 else 0
    rbf_cash_flow = (rbf_percentage * revenue) - operational_costs - rbf_repayment

    # Total cash flow combining all financing types
    total_net_cash_flow = equity_cash_flow + debt_cash_flow + rbf_cash_flow

    # Append data for the year
    complex_cashflow.append(
        [year, revenue, operational_costs, equity_cash_flow, debt_cash_flow, rbf_cash_flow, total_net_cash_flow]
    )

# Create a DataFrame to store simulation results
columns = [
    "Year",
    "Revenue ($M)",
    "Operational Costs ($M)",
    "Equity Cash Flow ($M)",
    "Debt Cash Flow ($M)",
    "RBF Cash Flow ($M)",
    "Total Net Cash Flow ($M)",
]
complex_df = pd.DataFrame(complex_cashflow, columns=columns)

# Plot cash flows for detailed comparison
plt.figure(figsize=(12, 8))

# Plot cash flows for each financing type
plt.plot(complex_df["Year"], complex_df["Equity Cash Flow ($M)"], marker='o', label="Equity Cash Flow", linewidth=2)
plt.plot(complex_df["Year"], complex_df["Debt Cash Flow ($M)"], marker='o', label="Debt Cash Flow", linewidth=2)
plt.plot(complex_df["Year"], complex_df["RBF Cash Flow ($M)"], marker='o', label="RBF Cash Flow", linewidth=2)
plt.plot(complex_df["Year"], complex_df["Total Net Cash Flow ($M)"], marker='s', label="Total Net Cash Flow", linewidth=2, linestyle='--')

# Adding titles and labels
plt.title("Cash Flow Comparison for Combined Financing Types", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Cash Flow ($M)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(years)
plt.legend(fontsize=12)
plt.tight_layout()

# Display the plot
plt.show()
