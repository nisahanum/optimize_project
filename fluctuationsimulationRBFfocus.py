import numpy as np
import matplotlib.pyplot as plt

# Number of simulations
num_simulations = 10000

# Define base assumptions for the simulation
# Parameters for costs of equity, debt, and RBF
cost_of_equity_mean = 0.12
cost_of_equity_std = 0.02

cost_of_debt_mean = 0.05
cost_of_debt_std = 0.01

rbf_cost_mean = 0.08  # Assume lower base RBF cost to make it flexible
rbf_cost_std = 0.015

# Parameters for increasing revenue (growing over time)
growth_rate_mean = 0.05  # 5% growth in revenue each period
growth_rate_std = 0.02

tax_rate_mean = 0.30
tax_rate_std = 0.05

# Proportions of each financing source (more RBF in this case)
equity_proportion_mean = 0.30
equity_proportion_std = 0.1

debt_proportion_mean = 0.30
debt_proportion_std = 0.1

rbf_proportion_mean = 0.40  # More RBF proportion to emphasize its flexibility
rbf_proportion_std = 0.05

# Simulating Revenue-Based Financing flexibility
def simulate_rbf_flexibility(num_simulations):
    total_wacc_values = []
    total_npv_values = []
    
    for _ in range(num_simulations):
        # Generate random values for cost of equity, debt, and RBF
        cost_of_equity = np.random.normal(cost_of_equity_mean, cost_of_equity_std)
        cost_of_debt = np.random.normal(cost_of_debt_mean, cost_of_debt_std)
        rbf_cost = np.random.normal(rbf_cost_mean, rbf_cost_std)
        tax_rate = np.random.normal(tax_rate_mean, tax_rate_std)

        # Generate random proportions for equity, debt, and RBF
        equity_proportion = np.clip(np.random.normal(equity_proportion_mean, equity_proportion_std), 0, 1)
        debt_proportion = np.clip(np.random.normal(debt_proportion_mean, debt_proportion_std), 0, 1)
        rbf_proportion = np.clip(np.random.normal(rbf_proportion_mean, rbf_proportion_std), 0, 1)

        # Normalize proportions to sum to 1
        total_proportion = equity_proportion + debt_proportion + rbf_proportion
        equity_proportion /= total_proportion
        debt_proportion /= total_proportion
        rbf_proportion /= total_proportion

        # Calculate the overall WACC with proportions of equity, debt, and RBF
        equity_contribution = equity_proportion * cost_of_equity
        debt_contribution = debt_proportion * cost_of_debt * (1 - tax_rate)
        rbf_contribution = rbf_proportion * rbf_cost

        total_wacc = equity_contribution + debt_contribution + rbf_contribution
        total_wacc_values.append(total_wacc)

        # Simulating revenue growth and calculating NPV with flexible RBF payments
        initial_revenue = 1000000  # Initial revenue in year 1
        npv = 0
        for year in range(1, 11):  # Assuming a 10-year project
            growth_rate = np.random.normal(growth_rate_mean, growth_rate_std)
            revenue = initial_revenue * ((1 + growth_rate) ** year)
            discount_factor = (1 + total_wacc) ** year
            npv += revenue / discount_factor

        total_npv_values.append(npv)

    return total_wacc_values, total_npv_values

# Running the simulation
wacc_simulations, npv_simulations = simulate_rbf_flexibility(num_simulations)

# Plot the results for WACC
plt.figure(figsize=(10, 6))
plt.hist(wacc_simulations, bins=50, alpha=0.7, color='purple', edgecolor='black')
plt.title('WACC Distribution - RBF vs. Equity vs. Debt')
plt.xlabel('WACC')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Plot the results for NPV
plt.figure(figsize=(10, 6))
plt.hist(npv_simulations, bins=50, alpha=0.7, color='green', edgecolor='black')
plt.title('NPV Distribution - RBF vs. Equity vs. Debt')
plt.xlabel('NPV')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Summary statistics
mean_wacc = np.mean(wacc_simulations)
std_wacc = np.std(wacc_simulations)
mean_npv = np.mean(npv_simulations)
std_npv = np.std(npv_simulations)

print(f"Mean WACC: {mean_wacc:.4f}")
print(f"Standard Deviation of WACC: {std_wacc:.4f}")
print(f"Mean NPV: {mean_npv:.2f}")
print(f"Standard Deviation of NPV: {std_npv:.2f}")
