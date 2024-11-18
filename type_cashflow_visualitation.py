import matplotlib.pyplot as plt

# Example Data
years = [1, 2, 3, 4, 5]  # Timeline (e.g., Years or Months)
cash_inflows = [500000, 600000, 720000, 864000, 1036800]  # Revenue inflows
cash_outflows = [200000, 300000, 350000, 400000, 450000]  # Expenses or loan repayments
net_cash_flows = [inflow - outflow for inflow, outflow in zip(cash_inflows, cash_outflows)]  # Net cash flow

# Plot the bar chart for inflows and outflows
plt.figure(figsize=(10, 6))
plt.bar(years, cash_inflows, color='green', alpha=0.7, label='Cash Inflows')
plt.bar(years, cash_outflows, color='red', alpha=0.7, label='Cash Outflows')

# Plot the line chart for net cash flow
plt.plot(years, net_cash_flows, color='blue', marker='o', linestyle='--', label='Net Cash Flow')

# Add labels and title
plt.title("Cash Flow Visualization: Inflows, Outflows, and Net", fontsize=14)
plt.xlabel("Time (Years)")
plt.ylabel("Cash Flow (USD)")
plt.xticks(years)
plt.axhline(y=0, color='black', linewidth=0.8, linestyle='--')  # Reference line at 0 for net cash flow
plt.legend()

# Display the chart
plt.tight_layout()
plt.show()
