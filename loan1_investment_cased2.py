# Simulating Loan 1 Financing for 5 Projects
import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt

# Input parameters for the simulation
num_projects = 5
initial_investments = [1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000]  # Rounded initial investments (USD)
interest_rates = [0.05, 0.06, 0.055, 0.065, 0.07]  # Annual interest rates for each project
loan_terms = [5, 5, 5, 5, 5]  # Loan term for each project (in years)
annual_revenues = [300_000, 500_000, 700_000, 900_000, 1_100_000]  # Annual revenues for each project

# Simulating cash flows and ROI for each project
loan_projects = []
loan_cash_flows = []

for i in range(num_projects):
    # Annual repayment calculation (principal + interest)
    principal = initial_investments[i] / loan_terms[i]
    outstanding_balance = initial_investments[i]
    repayments = []
    for year in range(1, loan_terms[i] + 1):
        interest_payment = outstanding_balance * interest_rates[i]
        total_payment = principal + interest_payment
        repayments.append(total_payment)
        outstanding_balance -= principal

    # Annual cash inflows and net cash flows
    net_cash_flows = [annual_revenues[i] - repayment for repayment in repayments]
    total_inflows = sum([annual_revenues[i]] * loan_terms[i])
    total_outflows = sum(repayments)
    net_cash_flow = total_inflows - total_outflows

    # ROI calculation
    roi = (net_cash_flow / initial_investments[i]) * 100

    # Append project summary
    loan_projects.append({
        "Project": f"Project {i + 1}",
        "Initial Investment (USD)": initial_investments[i],
        "Interest Rate (%)": interest_rates[i] * 100,
        "Loan Term (Years)": loan_terms[i],
        "Annual Revenue (USD)": annual_revenues[i],
        "Total Inflows (USD)": total_inflows,
        "Total Outflows (USD)": total_outflows,
        "Net Cash Flow (USD)": net_cash_flow,
        "ROI (%)": roi
    })

    # Append detailed cash flow for each project
    for year, repayment, net_flow in zip(range(1, loan_terms[i] + 1), repayments, net_cash_flows):
        loan_cash_flows.append({
            "Project": f"Project {i + 1}",
            "Year": year,
            "Cash Inflow (USD)": annual_revenues[i],
            "Repayment (USD)": repayment,
            "Net Cash Flow (USD)": net_flow
        })

# Create DataFrames
loan_summary = pd.DataFrame(loan_projects)
loan_cash_flow_details = pd.DataFrame(loan_cash_flows)

# Displaying the results
# Display DataFrames in the console
print("=== Loan 1 Project Portfolio Summary ===")
print(loan_summary)

print("\n=== Loan 1 Project Cash Flow Details ===")
print(loan_cash_flow_details)

# Save DataFrames to CSV files for further analysis
#loan_summary.to_csv("loan1_project_portfolio_summary.csv", index=False)
#loan_cash_flow_details.to_csv("loan1_project_cash_flow_details.csv", index=False)

#print("\nData saved to 'loan1_project_portfolio_summary.csv' and 'loan1_project_cash_flow_details.csv'.")


# Clear and combined visualization for inflows, outflows, and net cash flows


# Bar Chart for Total Inflows, Outflows, and Net Cash Flows
plt.figure(figsize=(12, 8))
bar_width = 0.3  # Adjusting bar width for better clarity
x_positions = range(len(loan_summary["Project"]))

# Plotting bars for each category
plt.bar(
    [pos - bar_width for pos in x_positions], 
    loan_summary["Total Inflows (USD)"], 
    width=bar_width, 
    label="Total Inflows", 
    alpha=0.8
)
plt.bar(
    x_positions, 
    loan_summary["Total Outflows (USD)"], 
    width=bar_width, 
    label="Total Outflows", 
    alpha=0.8
)
plt.bar(
    [pos + bar_width for pos in x_positions], 
    loan_summary["Net Cash Flow (USD)"], 
    width=bar_width, 
    label="Net Cash Flow", 
    alpha=0.8
)

# Adjusting labels and scales
plt.xticks(x_positions, loan_summary["Project"], rotation=45)
plt.title("Loan 1 Portfolio: Inflows, Outflows, and Net Cash Flow")
plt.xlabel("Projects")
plt.ylabel("USD")
plt.yscale("log")  # Applying logarithmic scale for better differentiation
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
