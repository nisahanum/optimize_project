# Simulating ROI for 5 Projects with Rounded Initial Investments
import pandas as pd  # For data manipulation
# Input parameters
num_projects = 5
initial_investments = [1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000]  # Rounded initial investments (USD)
roi_targets = [0.272, 0.3, 0.25, 0.28, 0.35]  # ROI targets for each project
revenue_growth_rate = 0.20  # Annual revenue growth rate (20%)
investment_horizon = 5  # Investment horizon in years

# Simulating cash flows and ROI for each project
portfolio_projects = []
portfolio_cash_flows = []

for i, investment in enumerate(initial_investments):
    # Base revenue calculation
    base_revenue = investment * roi_targets[i]

    # Annual cash inflows
    annual_inflows = [base_revenue * (1 + revenue_growth_rate) ** (year - 1) for year in range(1, investment_horizon + 1)]

    # Annual cash outflows
    annual_outflows = [investment if year == 1 else 0 for year in range(1, investment_horizon + 1)]

    # Annual net cash flows
    annual_net_cash_flows = [inflow - outflow for inflow, outflow in zip(annual_inflows, annual_outflows)]

    # Total inflows, outflows, and ROI
    total_inflows = sum(annual_inflows)
    total_outflows = sum(annual_outflows)
    net_cash_flow = total_inflows - total_outflows
    roi = (net_cash_flow / investment) * 100

    # Append project summary
    portfolio_projects.append({
        "Project": f"Project {i + 1}",
        "Initial Investment (USD)": investment,
        "ROI Target (%)": roi_targets[i] * 100,
        "Total Inflows (USD)": total_inflows,
        "Total Outflows (USD)": total_outflows,
        "Net Cash Flow (USD)": net_cash_flow,
        "ROI (%)": roi
    })

    # Append detailed cash flow for each project
    for year, inflow, outflow, net_flow in zip(range(1, investment_horizon + 1), annual_inflows, annual_outflows, annual_net_cash_flows):
        portfolio_cash_flows.append({
            "Project": f"Project {i + 1}",
            "Year": year,
            "Cash Inflow (USD)": inflow,
            "Cash Outflow (USD)": outflow,
            "Net Cash Flow (USD)": net_flow
        })

# Create DataFrames
portfolio_summary = pd.DataFrame(portfolio_projects)
portfolio_cash_flow_details = pd.DataFrame(portfolio_cash_flows)

# Displaying the results
# Display DataFrames in the console
print("=== Portfolio Project Summary ===")
print(portfolio_summary)

print("\n=== Portfolio Project Cash Flow Details ===")
print(portfolio_cash_flow_details)

# Save DataFrames to CSV files for further analysis
#portfolio_summary.to_csv("portfolio_project_summary.csv", index=False)
#portfolio_cash_flow_details.to_csv("portfolio_project_cash_flow_details.csv", index=False)

#print("\nData saved to 'portfolio_project_summary.csv' and 'portfolio_project_cash_flow_details.csv'.")

import matplotlib.pyplot as plt

# Visualization: Portfolio Summary - Total Inflows, Outflows, and Net Cash Flows
plt.figure(figsize=(12, 6))
x_labels = portfolio_summary["Project"]

# Bar Chart for Total Inflows, Outflows, and Net Cash Flows
plt.bar(x_labels, portfolio_summary["Total Inflows (USD)"], label="Total Inflows", alpha=0.7)
plt.bar(x_labels, portfolio_summary["Total Outflows (USD)"], label="Total Outflows", alpha=0.7)
plt.bar(x_labels, portfolio_summary["Net Cash Flow (USD)"], label="Net Cash Flow", alpha=0.7)

# Adding labels and legend
plt.title("Portfolio Summary: Inflows, Outflows, and Net Cash Flows")
plt.xlabel("Projects")
plt.ylabel("USD")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Visualization: Detailed Cash Flow for Each Project Over the Horizon
plt.figure(figsize=(14, 8))

for project in portfolio_summary["Project"]:
    project_data = portfolio_cash_flow_details[portfolio_cash_flow_details["Project"] == project]
    plt.plot(
        project_data["Year"],
        project_data["Net Cash Flow (USD)"],
        marker="o",
        label=project,
    )

# Adding labels and legend
plt.title("Net Cash Flow Over Time for Each Project")
plt.xlabel("Year")
plt.ylabel("Net Cash Flow (USD)")
plt.legend(title="Projects")
plt.grid(True)
plt.tight_layout()
plt.show()

